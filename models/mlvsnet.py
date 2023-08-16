""" 
MLVSNet.py
Created by QW at 2022/10/11 15:32
"""

from copy import deepcopy
from torch import nn
import torch
from models.backbone.pointnet import Pointnet_Backbone
from models.head.xcorr import TGA
from models.head.rpn import MLVSVoteNetRPN
from models import base_model
from pointnet2.utils import pytorch_utils as pt_utils
from pointnet2.utils.pointnet2_modules import PointnetSAModule
import torch.nn.functional as F


class MLVSNET(base_model.MatchingBaseModel):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.save_hyperparameters()
        self.backbone = Pointnet_Backbone(self.config.use_fps, self.config.normalize_xyz, return_intermediate=True)
        self.conv_final = nn.Conv1d(256, self.config.feature_channel, kernel_size=1)

        self.tga2 = TGA(t_feature_channel=self.config.feature_channel,
                               s_feature_channel=self.config.feature_channel)
        self.tga3 = TGA(t_feature_channel=self.config.feature_channel,
                               s_feature_channel=self.config.feature_channel)

        self.FC_layer_cla_2 = (
            pt_utils.Seq(self.config.feature_channel)
                .conv1d(self.config.feature_channel, bn=True)
                .conv1d(self.config.feature_channel, bn=True)
                .conv1d(1, activation=None))

        self.trans_layer_2 = PointnetSAModule(
            radius=0.3,
            nsample=32,
            mlp=[256, 256],
            use_xyz=True)

        self.FC_layer_cla = (
            pt_utils.Seq(self.config.feature_channel)
                .conv1d(self.config.feature_channel, bn=True)
                .conv1d(self.config.feature_channel, bn=True)
                .conv1d(1, activation=None))


        self.vote_layer_2 = (
            pt_utils.Seq(3 + self.config.feature_channel)
                .conv1d(self.config.feature_channel, bn=True)
                .conv1d(self.config.feature_channel, bn=True)
                .conv1d(3 + self.config.feature_channel, activation=None))
        self.vote_layer = (
            pt_utils.Seq(3 + self.config.feature_channel)
                .conv1d(self.config.feature_channel, bn=True)
                .conv1d(self.config.feature_channel, bn=True)
                .conv1d(3 + self.config.feature_channel, activation=None))

        self.rpn = MLVSVoteNetRPN(self.config.feature_channel,
                                 vote_channel=self.config.vote_channel,
                                 num_proposal=self.config.num_proposal,
                                 normalize_xyz=self.config.normalize_xyz)

    def compute_loss(self, data, output,mix_rate=None):
        # out_dict = super(CYCLEBAT, self).compute_loss(data, output)
        estimation_boxes = output['estimation_boxes']  # B,num_proposal,5
        estimation_cla = output['estimation_cla']  # B,N

        seg_label = data['seg_label']

        box_label = data['box_label']  # B,4
        proposal_center = output["center_xyz"]  # B,num_proposal,3
        vote_xyz = output["vote_xyz"]

        loss_seg_0 = F.binary_cross_entropy_with_logits(estimation_cla[0], seg_label)
        loss_seg_1 = F.binary_cross_entropy_with_logits(estimation_cla[1], seg_label)

        loss_vote_0 = F.smooth_l1_loss(vote_xyz[0], box_label[:, None, :3].expand_as(vote_xyz[0]), reduction='none')  # B,N,3
        loss_vote_0 = (loss_vote_0.mean(2) * seg_label).sum() / (seg_label.sum() + 1e-06)

        loss_vote_1 = F.smooth_l1_loss(vote_xyz[1], box_label[:, None, :3].expand_as(vote_xyz[1]), reduction='none')  # B,N,3
        loss_vote_1 = (loss_vote_1.mean(2) * seg_label).sum() / (seg_label.sum() + 1e-06)

        dist = torch.sum((proposal_center - box_label[:, None, :3]) ** 2, dim=-1)

        dist = torch.sqrt(dist + 1e-6)  # B, K
        objectness_label = torch.zeros_like(dist, dtype=torch.float)
        objectness_label[dist < 0.3] = 1.0
        objectness_score = estimation_boxes[:, :, 4]  # B, K
        objectness_mask = torch.zeros_like(objectness_label, dtype=torch.float)
        objectness_mask[dist < 0.3] = 1
        objectness_mask[dist > 0.6] = 1
        loss_objective = F.binary_cross_entropy_with_logits(objectness_score, objectness_label,
                                                            pos_weight=torch.tensor([2.0]).cuda())
        loss_objective = torch.sum(loss_objective * objectness_mask) / (
                torch.sum(objectness_mask) + 1e-6)
        loss_box = F.smooth_l1_loss(estimation_boxes[:, :, :4],
                                    box_label[:, None, :4].expand_as(estimation_boxes[:, :, :4]),
                                    reduction='none')
        loss_box = torch.sum(loss_box.mean(2) * objectness_label) / (objectness_label.sum() + 1e-6)

        # out_dict["loss_bc"] = loss_bc
        return {
            "loss_objective": loss_objective,
            "loss_box": loss_box,
            "loss_seg_0": loss_seg_0,
            "loss_seg_1": loss_seg_1,
            "loss_vote_0": loss_vote_0,
            "loss_vote_1": loss_vote_1,
        }

    def forward(self, input_dict):
        """
        :param input_dict:
        {
        'template_points': template_points.astype('float32'),
        'search_points': search_points.astype('float32'),
        'box_label': np.array(search_bbox_reg).astype('float32'),
        'bbox_size': search_box.wlh,
        'seg_label': seg_label.astype('float32'),
        }

        :return:
        """
        template = input_dict['template_points']
        search = input_dict['search_points']
        M = template.shape[1]
        N = search.shape[1]
        template_xyz, template_feature, _ = self.backbone(template, [M // 2, M // 4, M // 8])
        search_xyz, search_feature, sample_idxs = self.backbone(search, [N // 2, N // 4, N // 8])
        search_feature[2] = self.conv_final(search_feature[2])
        new_search_feature_2 = self.tga2(search_feature[1], template_feature[1])
        fusion_feature = self.tga3(search_feature[2], template_feature[2])

        v_2 = self.trans_layer_2(search_xyz[1], new_search_feature_2, 128)

        # layer_2
        estimation_cla_2 = self.FC_layer_cla_2(v_2[1]).squeeze(1)
        score_2 = estimation_cla_2.sigmoid()
        fusion_xyz_feature_2 = torch.cat((v_2[0].transpose(1, 2).contiguous(), v_2[1]), dim=1)
        offset_2 = self.vote_layer_2(fusion_xyz_feature_2)
        vote_2 = fusion_xyz_feature_2 + offset_2
        vote_xyz_2 = vote_2[:, 0:3, :].transpose(1, 2).contiguous()
        vote_feature_2 = vote_2[:, 3:, :]
        vote_feature_2 = torch.cat((score_2.unsqueeze(1), vote_feature_2), dim=1)

        #layer 3
        estimation_cla = self.FC_layer_cla(fusion_feature).squeeze(1)
        score = estimation_cla.sigmoid()
        fusion_xyz_feature = torch.cat((search_xyz[2].transpose(1, 2).contiguous(), fusion_feature), dim=1)
        offset = self.vote_layer(fusion_xyz_feature)
        vote = fusion_xyz_feature + offset
        vote_xyz = vote[:, 0:3, :].transpose(1, 2).contiguous()
        vote_feature = vote[:, 3:, :]
        vote_feature = torch.cat((score.unsqueeze(1), vote_feature), dim=1)

        # concat
        estimation_cla_s = [estimation_cla_2, estimation_cla]
        vote_xyz_s = [vote_xyz_2, vote_xyz]
        vote_xyz_cat = torch.cat([vote_xyz_2, vote_xyz], 1)
        vote_feature_cat = torch.cat([vote_feature_2, vote_feature], 2)
        estimation_boxes, center_xyzs = self.rpn(vote_xyz_cat, vote_feature_cat)
        end_points = {"estimation_boxes": estimation_boxes,
                      "vote_center": vote_xyz_s,
                      "pred_seg_score": estimation_cla_s,
                      "center_xyz": center_xyzs,
                      'sample_idxs': sample_idxs,
                      'estimation_cla': estimation_cla_s,
                      "vote_xyz": vote_xyz_s,
                      }
        return end_points


    def training_step(self, batch, batch_idx):
        """
        {"estimation_boxes": estimation_boxs.transpose(1, 2).contiguous(),
                  "vote_center": vote_xyz,
                  "pred_seg_score": estimation_cla,
                  "center_xyz": center_xyzs,
                  "seed_idxs":
                  "seg_label"
        }
        """
        end_points = self(batch)
        estimation_cla = end_points['estimation_cla']  # B,N
        N = estimation_cla[1].shape[1]
        seg_label = batch['seg_label']
        sample_idxs = end_points['sample_idxs']  # B,N
        # update label
        seg_label = seg_label.gather(dim=1, index=sample_idxs[:, :N].long())  # B,N
        estimation_cla = end_points['estimation_cla'][0]  # B,N
        N = estimation_cla.shape[1]
        seg_label = batch['seg_label']
        sample_idxs = end_points['sample_idxs']  # B,N
        seg_label = seg_label.gather(dim=1, index=sample_idxs[:, :N].long())  # B,N
        batch['seg_label'] = seg_label
        # compute loss
        loss_dict = self.compute_loss(batch, end_points)
        loss = loss_dict['loss_objective'] * self.config.objectiveness_weight \
               + loss_dict['loss_box'] * self.config.box_weight \
               + loss_dict['loss_seg_0'] * self.config.seg_weight \
               + loss_dict['loss_vote_0'] * self.config.vote_weight \
               + loss_dict['loss_seg_1'] * self.config.seg_weight \
               + loss_dict['loss_vote_1'] * self.config.vote_weight
        self.log('loss/train', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log('loss_box/train', loss_dict['loss_box'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)
        self.log('loss_seg/train', loss_dict['loss_seg_0'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)
        self.log('loss_seg/train', loss_dict['loss_seg_1'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)
        self.log('loss_vote/train', loss_dict['loss_vote_0'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)
        self.log('loss_vote/train', loss_dict['loss_vote_1'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)
        self.log('loss_objective/train', loss_dict['loss_objective'].item(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=False)
        self.logger.experiment.add_scalars('loss', {'loss_total': loss.item(),
                                                    'loss_box': loss_dict['loss_box'].item(),
                                                    'loss_seg_0': loss_dict['loss_seg_0'].item(),
                                                    'loss_seg_1': loss_dict['loss_seg_1'].item(),
                                                    'loss_vote_1': loss_dict['loss_vote_0'].item(),
                                                    'loss_vote_1': loss_dict['loss_vote_1'].item(),
                                                    'loss_objective': loss_dict['loss_objective'].item()},
                                           global_step=self.global_step)

        return loss
