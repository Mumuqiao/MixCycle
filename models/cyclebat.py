""" 
cycbat.py
Created by qw at 2022/5/26
"""

from copy import deepcopy
from cv2 import FileStorage_READ
import torch
from torch import nn
from datasets.searchspace import Filerts
from models.backbone.pointnet import Pointnet_Backbone
from models.head.xcorr import BoxAwareXCorr
from models.head.rpn import P2BVoteNetRPN
from models import base_model
import torch.nn.functional as F
from datasets import points_utils
from pointnet2.utils import pytorch_utils as pt_utils
import numpy as np


class CYCLEBAT(base_model.MatchingBaseModel):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.save_hyperparameters()
        self.backbone = Pointnet_Backbone(self.config.use_fps, self.config.normalize_xyz, return_intermediate=False)
        self.conv_final = nn.Conv1d(256, self.config.feature_channel, kernel_size=1)
        self.mlp_bc = (pt_utils.Seq(3 + self.config.feature_channel)
                       .conv1d(self.config.feature_channel, bn=True)
                       .conv1d(self.config.feature_channel, bn=True)
                       .conv1d(self.config.bc_channel, activation=None))

        self.xcorr = BoxAwareXCorr(feature_channel=self.config.feature_channel,
                                   hidden_channel=self.config.hidden_channel,
                                   out_channel=self.config.out_channel,
                                   k=self.config.k,
                                   use_search_bc=self.config.use_search_bc,
                                   use_search_feature=self.config.use_search_feature,
                                   bc_channel=self.config.bc_channel)
        self.rpn = P2BVoteNetRPN(self.config.feature_channel,
                                 vote_channel=self.config.vote_channel,
                                 num_proposal=self.config.num_proposal,
                                 normalize_xyz=self.config.normalize_xyz)


    def prepare_input(self, template_pc, search_pc, template_box):
        template_points, idx_t = points_utils.regularize_pc(template_pc.points.T, self.config.template_size,
                                                            seed=1)
        search_points, idx_s = points_utils.regularize_pc(search_pc.points.T, self.config.search_size,
                                                          seed=1)
        template_points_torch = torch.tensor(template_points, device=self.device, dtype=torch.float32,requires_grad=False)
        search_points_torch = torch.tensor(search_points, device=self.device, dtype=torch.float32,requires_grad=False)
        template_bc = points_utils.get_point_to_box_distance(template_points, template_box)
        template_bc_torch = torch.tensor(template_bc, device=self.device, dtype=torch.float32,requires_grad=False)
        data_dict = {
            'template_points': template_points_torch[None, ...],
            'search_points': search_points_torch[None, ...],
            'points2cc_dist_t': template_bc_torch[None, ...],
            'idx_s': idx_s,
        }
        return data_dict

    def compute_loss(self, data, output,mix_rate=None):
        # out_dict = super(CYCLEBAT, self).compute_loss(data, output)
        estimation_boxes = output['estimation_boxes']  # B,num_proposal,5
        estimation_cla = output['estimation_cla']  # B,N
        if mix_rate != None:
            tmix_rate = deepcopy(mix_rate)
            seg_label = data['seg_label'] * tmix_rate.unsqueeze(1).expand_as(data['seg_label'])
        else:
            seg_label = data['seg_label']
        box_label = data['box_label']  # B,4
        proposal_center = output["center_xyz"]  # B,num_proposal,3
        vote_xyz = output["vote_xyz"]

        loss_seg = F.binary_cross_entropy_with_logits(estimation_cla, seg_label)

        loss_vote = F.smooth_l1_loss(vote_xyz, box_label[:, None, :3].expand_as(vote_xyz), reduction='none')  # B,N,3
        loss_vote = (loss_vote.mean(2) * seg_label).sum() / (seg_label.sum() + 1e-06)

        dist = torch.sum((proposal_center - box_label[:, None, :3]) ** 2, dim=-1)

        dist = torch.sqrt(dist + 1e-6)  # B, K
        objectness_label = torch.zeros_like(dist, dtype=torch.float)
        if mix_rate is not None:
            tmix_rate = deepcopy(mix_rate)
            tmix_rate = tmix_rate.unsqueeze(1).expand_as(objectness_label)
            objectness_label[dist < 0.3] = 1.0 * tmix_rate[dist < 0.3]
        else:
            objectness_label[dist < 0.3] = 1.0
        if data['objectness_label'] is not None:
            objectness_label = objectness_label * data['objectness_label'].unsqueeze(1).expand_as(objectness_label)
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

        search_bc = data['points2cc_dist_s']
        pred_search_bc = output['pred_search_bc']
        loss_bc = F.smooth_l1_loss(pred_search_bc, search_bc, reduction='none')
        loss_bc = torch.sum(loss_bc.mean(2) * seg_label) / (seg_label.sum() + 1e-6)
        # out_dict["loss_bc"] = loss_bc
        return {
            "loss_objective": loss_objective,
            "loss_box": loss_box,
            "loss_seg": loss_seg,
            "loss_vote": loss_vote,
            "loss_bc": loss_bc,
        }

    def compute_backward_loss(self, data, output,mix_rate = None):
        estimation_boxes = output['estimation_boxes']  # B,num_proposal,5
        box_label = data['box_label']  # B,4
        proposal_center = output["center_xyz"]  # B,num_proposal,3
        
        dist = torch.sum((proposal_center - box_label[:, None, :3]) ** 2, dim=-1)

        dist = torch.sqrt(dist + 1e-6)  # B, K
        objectness_label = torch.zeros_like(dist, dtype=torch.float)
        if mix_rate is not None:
            tmix_rate = deepcopy(mix_rate)
            tmix_rate = tmix_rate.unsqueeze(1).expand_as(objectness_label)
            objectness_label[dist < 0.3] = 1.0 * tmix_rate[dist < 0.3]
        else:
            objectness_label[dist < 0.3] = 1
        # if data['objectness_label'] is not None:
        #     objectness_label = objectness_label * data['objectness_label'].unsqueeze(1).expand_as(objectness_label)
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

        return {
            "loss_objective": loss_objective,
            "loss_box": loss_box,
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
        'points2cc_dist_t': template_bc,
        'points2cc_dist_s': search_bc,
        }

        :return:
        """
        template = input_dict['template_points']
        search = input_dict['search_points']
        template_bc = input_dict['points2cc_dist_t']
        M = template.shape[1]
        N = search.shape[1]

        # backbone
        template_xyz, template_feature, sample_idxs_t = self.backbone(template, [M // 2, M // 4, M // 8])
        search_xyz, search_feature, sample_idxs = self.backbone(search, [N // 2, N // 4, N // 8])
        template_feature = self.conv_final(template_feature)
        search_feature = self.conv_final(search_feature)
        # prepare bc
        pred_search_bc = self.mlp_bc(torch.cat([search_xyz.transpose(1, 2), search_feature], dim=1))  # B, 9, N // 8
        pred_search_bc = pred_search_bc.transpose(1, 2)
        sample_idxs_t = sample_idxs_t[:, :M // 8, None]
        template_bc = template_bc.gather(dim=1, index=sample_idxs_t.repeat(1, 1, self.config.bc_channel).long())
        # box-aware xcorr
        fusion_feature = self.xcorr(template_feature, search_feature, template_xyz, search_xyz, template_bc,
                                    pred_search_bc)
        # proposal generation
        estimation_boxes, estimation_cla, vote_xyz, center_xyzs = self.rpn(search_xyz, fusion_feature)
        end_points = {"estimation_boxes": estimation_boxes,
                      "vote_center": vote_xyz,
                      "pred_seg_score": estimation_cla,
                      "center_xyz": center_xyzs,
                      'sample_idxs': sample_idxs,
                      'estimation_cla': estimation_cla,
                      "vote_xyz": vote_xyz,
                      "pred_search_bc": pred_search_bc
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
                  "pred_search_bc": pred_search_bc
        }
        """
        loss = 0
        batch.update({'objectness_label': None})
        end_points = self(batch)
        search_pc = batch['points2cc_dist_s']
        estimation_cla = end_points['estimation_cla']  # B,N
        N = estimation_cla.shape[1]
        seg_label = batch['seg_label']
        sample_idxs = end_points['sample_idxs']  # B,N
        seg_label = seg_label.gather(dim=1, index=sample_idxs[:, :N].long())  # B,N
        search_pc = search_pc.gather(dim=1, index=sample_idxs[:, :N, None].repeat(1, 1, self.config.bc_channel).long())
        # update label
        batchself = deepcopy(batch)
        batchself['seg_label'] = seg_label
        batchself['points2cc_dist_s'] = search_pc
        # compute loss
        loss_dict = self.compute_loss(batchself, end_points,batchself['mix_rate'][:,0])

        loss = loss_dict['loss_objective'] * self.config.objectiveness_weight \
            + loss_dict['loss_box'] * self.config.box_weight
        loss += loss_dict['loss_seg'] * self.config.seg_weight \
            + loss_dict['loss_vote'] * self.config.vote_weight \
            + loss_dict['loss_bc'] * self.config.bc_weight


        #forward 1
        with torch.no_grad():
            forward1_data_dict = {
            'template_points': [],
            'search_points': [],
            'points2cc_dist_t': [],
            'box_label': [],
            'bbox_size': [],
            'seg_label': [],
            'points2cc_dist_s': [],
            'idx_s': [],
            'search_pc_crop': [],
            'objectness_label': None,
            }
            for template_frame,forward1_frame in zip(batch['template_frame'],batch['forward1_frame']):
                template_box = template_frame['3d_bbox']
                results_bbs = [template_box,template_box]
                search_pc_crop, ref_bb = self.generate_search_area([forward1_frame],0,results_bbs)
                model_pc, model_box = self.generate_template([template_frame,template_frame], 1, results_bbs)
                # print(search_pc_crop.points.shape)
                temp_dict = self.prepare_input(model_pc,search_pc_crop,model_box)
                # print('xxxxxxxx',search_pc_crop.points.shape)
                forward1_data_dict['template_points'] += [temp_dict['template_points']]
                forward1_data_dict['search_points'] += [temp_dict['search_points']]
                forward1_data_dict['points2cc_dist_t'] += [temp_dict['points2cc_dist_t']]
                forward1_data_dict['idx_s'] += [temp_dict['idx_s']]
                forward1_data_dict['search_pc_crop'] += [search_pc_crop]
            forward1_data_dict['template_points'] = torch.cat(forward1_data_dict['template_points'])
            forward1_data_dict['search_points'] = torch.cat(forward1_data_dict['search_points'])
            forward1_data_dict['points2cc_dist_t'] = torch.cat(forward1_data_dict['points2cc_dist_t'])
            # forward1_data_dict['idx_s'] = np.stack(forward1_data_dict['idx_s'])
       
        with torch.no_grad():
            forward1_end_points = self(forward1_data_dict)
       
        
        with torch.no_grad():
            estimation_box = forward1_end_points['estimation_boxes']
            estimation_boxes = estimation_box.detach()
            estimation_boxes_cpu = estimation_boxes.cpu().numpy()
            best_box_idx = estimation_boxes_cpu[:,:,4].argmax(axis=1)
            backward1_object_label = estimation_boxes[range(len(best_box_idx)),best_box_idx,4]
            estimation_box_cpu = estimation_boxes_cpu[range(len(best_box_idx)),best_box_idx,0:4]
            candidate_degrees_1 = deepcopy(estimation_box_cpu[:,3]) 
            
            candidate_box_1 = [points_utils.getOffsetBB(refbox['3d_bbox'],esoffset, degrees=self.config.degrees,
                                                    use_z=self.config.use_z,
                                                    limit_box=self.config.limit_box) for refbox, esoffset in zip(batch['template_frame'], estimation_box_cpu)]

            #Use if forward2 
            forward2_data_dict = {
            'template_points': [],
            'search_points': [],
            'points2cc_dist_t': [],
            'box_label': [],
            'bbox_size': [],
            'seg_label': [],
            'points2cc_dist_s': [],
            'objectness_label': None,
            }
            for template_frame,forward1_frame,forward2_frame,canbox in zip(batch['template_frame'],
                        batch['forward1_frame'],batch['forward2_frame'],candidate_box_1):
                template_box = template_frame['3d_bbox']
                results_bbs= [template_box,canbox,template_box]
                search_pc_crop, ref_bb = self.generate_search_area([forward2_frame],0,results_bbs)
                model_pc, model_box = self.generate_template([template_frame,forward1_frame], 2, results_bbs)
                temp_dict = self.prepare_input(model_pc,search_pc_crop,model_box)
                forward2_data_dict['template_points'] += [temp_dict['template_points']]
                forward2_data_dict['search_points'] += [temp_dict['search_points']]
                forward2_data_dict['points2cc_dist_t'] += [temp_dict['points2cc_dist_t']]
            forward2_data_dict['template_points'] = torch.cat(forward2_data_dict['template_points'])
            forward2_data_dict['search_points'] = torch.cat(forward2_data_dict['search_points'])
            forward2_data_dict['points2cc_dist_t'] = torch.cat(forward2_data_dict['points2cc_dist_t'])
            

        #forward 2
        with torch.no_grad():
            forward2_end_points = self(forward2_data_dict)
            estimation_box = forward2_end_points['estimation_boxes']
            estimation_boxes_cpu = estimation_box.detach().cpu().numpy()
            best_box_idx = estimation_boxes_cpu[:,:,4].argmax(axis=1)
            estimation_box_cpu = estimation_boxes_cpu[range(len(best_box_idx)),best_box_idx,0:4]
            candidate_degrees_2 = deepcopy(estimation_box_cpu[:,3])
            candidate_box_2 = [points_utils.getOffsetBB(refbox,esoffset, degrees=self.config.degrees,
                                                    use_z=self.config.use_z,
                                                    limit_box=self.config.limit_box) for refbox, esoffset in zip(candidate_box_1, estimation_box_cpu)]
            
            backward1_data_dict = {
            'template_points': [],
            'search_points': [],
            'points2cc_dist_t': [],
            'box_label': [],
            'bbox_size': [],
            'seg_label': [],
            'points2cc_dist_s': [],
            'objectness_label': backward1_object_label,
            }
            ref_candidate_box_2 = []
            candidate_degrees_2 += batch['backward_offset'][:,0,2]
            count = 0
            for template_frame,forward2_frame,backward_frame1,canbox2,canbox1,backward_offset in zip(batch['template_frame'],
                        batch['forward2_frame'],batch['backward_frame1'],candidate_box_2,candidate_box_1,batch['backward_offset'] ):
                backward_offset = backward_offset[0]
                template_box = template_frame['3d_bbox']
                results_bbs= [template_box,canbox2]
                model_pc, model_box = self.generate_template([template_frame,forward2_frame], 2, results_bbs)
                ref_canbox2 = points_utils.getOffsetBB(canbox2, backward_offset,limit_box=self.config.limit_box, degrees=self.config.degrees)
                results_bbs[1] = ref_canbox2
                ref_candidate_box_2 += [ref_canbox2]
                search_pc_crop, ref_bb = self.generate_search_area([backward_frame1],0,results_bbs)
                temp_dict = self.prepare_input(model_pc,search_pc_crop,model_box)
                backward1_data_dict['template_points'] += [temp_dict['template_points']]
                backward1_data_dict['search_points'] += [temp_dict['search_points']]
                backward1_data_dict['points2cc_dist_t'] += [temp_dict['points2cc_dist_t']]
                #Genelize supervision information for backward
                backward_box1 = points_utils.transform_box(canbox1,ref_canbox2)
                backward1_data_dict['box_label'] += [np.array([backward_box1.center[0],backward_box1.center[1],backward_box1.center[2],-candidate_degrees_2[count]])]
                count += 1
                
                backward1_data_dict['bbox_size'] += [backward_box1.wlh]
                seg_label = points_utils.get_in_box_mask(search_pc_crop, backward_box1)

                if temp_dict['idx_s'] is None:
                    seg_label = np.zeros((1024,),dtype=np.float32)
                else:
                    seg_label = seg_label[temp_dict['idx_s']]
                # print(seg_label.shape)
                backward1_data_dict['seg_label'] += [seg_label.astype('float32')]
                search_bc = points_utils.get_point_to_box_distance(temp_dict['search_points'][0].cpu().numpy(),backward_box1)
                backward1_data_dict['points2cc_dist_s'] += [search_bc.astype('float32')]
            backward1_data_dict['template_points'] = torch.cat(backward1_data_dict['template_points'])
            backward1_data_dict['search_points'] = torch.cat(backward1_data_dict['search_points'])
            backward1_data_dict['points2cc_dist_t'] = torch.cat(backward1_data_dict['points2cc_dist_t'])
            backward1_data_dict['box_label'] = np.stack(backward1_data_dict['box_label'],axis=0)
            backward1_data_dict['bbox_size'] = np.stack(backward1_data_dict['bbox_size'],axis=0)
            backward1_data_dict['seg_label'] = np.stack(backward1_data_dict['seg_label'],axis=0)
            backward1_data_dict['points2cc_dist_s'] = np.stack(backward1_data_dict['points2cc_dist_s'],axis=0)
            backward1_data_dict['box_label'] = torch.from_numpy(backward1_data_dict['box_label'].astype('float32')).cuda()
            backward1_data_dict['bbox_size'] = torch.from_numpy(backward1_data_dict['bbox_size'].astype('float32')).cuda()
            backward1_data_dict['seg_label'] = torch.from_numpy(backward1_data_dict['seg_label']).cuda()
            backward1_data_dict['points2cc_dist_s'] = torch.from_numpy(backward1_data_dict['points2cc_dist_s']).cuda()

        #backward1

        with torch.no_grad():
            backward1_end_point = self(backward1_data_dict)
       

                   
        estimation_box = backward1_end_point['estimation_boxes']
        estimation_boxes_cpu = estimation_box.detach().cpu().numpy()
        best_box_idx = estimation_boxes_cpu[:,:,4].argmax(axis=1)
        estimation_box_cpu = estimation_boxes_cpu[range(len(best_box_idx)),best_box_idx,0:4]
        backward_candidate_box_1 = [points_utils.getOffsetBB(refbox,esoffset, degrees=self.config.degrees,
                                                use_z=self.config.use_z,
                                                limit_box=self.config.limit_box) for refbox, esoffset in zip(ref_candidate_box_2, estimation_box_cpu)]


    
        with torch.no_grad():
            backward2_data_dict = {
            'template_points': [],
            'search_points': [],
            'points2cc_dist_t': [],
            'box_label': [],
            'bbox_size': [],
            'seg_label': [],
            'points2cc_dist_s': [],
            'objectness_label': None,
            }
            candidate_degrees_1 += batch['backward_offset'][:,1,2]
            count = 0
            for template_frame,backward_frame1,backward_template_frame,canbox2, backward_offset in zip(batch['template_frame'],
                    batch['backward_frame1'],batch['backward_template_frame'],backward_candidate_box_1, batch['backward_offset']):
                backward_offset = backward_offset[1]
                template_box = template_frame['3d_bbox']
                results_bbs= [template_box,canbox2]
                model_pc, model_box = self.generate_template([template_frame,backward_frame1], 2, results_bbs)
                ref_canbox2 = points_utils.getOffsetBB(canbox2, backward_offset,limit_box=self.config.limit_box, degrees=self.config.degrees)
                results_bbs[1] = ref_canbox2
                search_pc_crop, ref_bb = self.generate_search_area([backward_template_frame],0,results_bbs)
                temp_dict = self.prepare_input(model_pc,search_pc_crop,model_box)
                backward2_data_dict['template_points'] += [temp_dict['template_points']]
                backward2_data_dict['search_points'] += [temp_dict['search_points']]
                backward2_data_dict['points2cc_dist_t'] += [temp_dict['points2cc_dist_t']]
                #Genelize supervision information for backward
                canbox1 = deepcopy(backward_template_frame['3d_bbox'])
                backward_box1 = points_utils.transform_box(canbox1,ref_canbox2)
                backward2_data_dict['box_label'] += [np.array([backward_box1.center[0],backward_box1.center[1],backward_box1.center[2],-candidate_degrees_1[count]])]
                count += 1
                # print(backward_box1.orientation.degrees)
                backward2_data_dict['bbox_size'] += [backward_box1.wlh]
                seg_label = points_utils.get_in_box_mask(search_pc_crop, backward_box1)
                if temp_dict['idx_s'] is None:
                    seg_label = np.zeros((1024,),dtype=np.float32)
                else:
                    seg_label = seg_label[temp_dict['idx_s']]
                # seg_label = seg_label[temp_dict['idx_s']]
                backward2_data_dict['seg_label'] += [seg_label.astype('float32')]
                search_bc = points_utils.get_point_to_box_distance(temp_dict['search_points'][0].cpu().numpy(),backward_box1)
                backward2_data_dict['points2cc_dist_s'] += [search_bc.astype('float32')]
            backward2_data_dict['template_points'] = torch.cat(backward2_data_dict['template_points'])
            backward2_data_dict['search_points'] = torch.cat(backward2_data_dict['search_points'])
            backward2_data_dict['points2cc_dist_t'] = torch.cat(backward2_data_dict['points2cc_dist_t'])
            backward2_data_dict['box_label'] = np.stack(backward2_data_dict['box_label'],axis=0)
            backward2_data_dict['bbox_size'] = np.stack(backward2_data_dict['bbox_size'],axis=0)
            backward2_data_dict['seg_label'] = np.stack(backward2_data_dict['seg_label'],axis=0)
            backward2_data_dict['points2cc_dist_s'] = np.stack(backward2_data_dict['points2cc_dist_s'],axis=0)
            backward2_data_dict['box_label'] = torch.from_numpy(backward2_data_dict['box_label'].astype('float32')).cuda()
            backward2_data_dict['bbox_size'] = torch.from_numpy(backward2_data_dict['bbox_size'].astype('float32')).cuda()
            backward2_data_dict['seg_label'] = torch.from_numpy(backward2_data_dict['seg_label'].astype('float32')).cuda()
            backward2_data_dict['points2cc_dist_s'] = torch.from_numpy(backward2_data_dict['points2cc_dist_s'].astype('float32')).cuda()

        #Backward 2
        backward2_end_point = self(backward2_data_dict)
        search_pc = backward2_data_dict['points2cc_dist_s']
        estimation_cla = backward2_end_point['estimation_cla']  # B,N
        N = estimation_cla.shape[1]
        seg_label = backward2_data_dict['seg_label']
        sample_idxs = backward2_end_point['sample_idxs']  # B,N
        seg_label = seg_label.gather(dim=1, index=sample_idxs[:, :N].long())  # B,N
        search_pc = search_pc.gather(dim=1, index=sample_idxs[:, :N, None].repeat(1, 1, self.config.bc_channel).long())
        # update label
        backward2_data_dict['seg_label'] = seg_label
        backward2_data_dict['points2cc_dist_s'] = search_pc
        # compute loss
        loss_dict_cyc = self.compute_loss(backward2_data_dict, backward2_end_point,batch["mix_rate"][:,1])
        loss += loss_dict_cyc['loss_objective'] * self.config.objectiveness_weight_backward \
            + loss_dict_cyc['loss_box'] * self.config.box_weight_backward \
            + loss_dict_cyc['loss_seg'] * self.config.seg_weight_backward \
            + loss_dict_cyc['loss_vote'] * self.config.vote_weight_backward \
            + loss_dict_cyc['loss_bc'] * self.config.bc_weight_backward

        # log
        self.log('loss/train', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log('loss_box/train', loss_dict['loss_box'].item(), on_step=True, on_epoch=True, prog_bar=True,
                logger=False)
        self.log('loss_seg/train', loss_dict['loss_seg'].item(), on_step=True, on_epoch=True, prog_bar=True,
                logger=False)
        self.log('loss_vote/train', loss_dict['loss_vote'].item(), on_step=True, on_epoch=True, prog_bar=True,
                logger=False)
        self.log('loss_bc/train', loss_dict['loss_bc'].item(), on_step=True, on_epoch=True, prog_bar=True,
                logger=False)
        self.log('loss_objective/train', loss_dict['loss_objective'].item(), on_step=True, on_epoch=True, prog_bar=True,
                logger=False)
        self.log('loss_box_cyc/train', loss_dict_cyc['loss_box'].item(), on_step=True, on_epoch=True, prog_bar=True,
                logger=False)
        self.log('loss_seg_cyc/train', loss_dict_cyc['loss_seg'].item(), on_step=True, on_epoch=True, prog_bar=True,
                logger=False)
        self.log('loss_vote_cyc/train', loss_dict_cyc['loss_vote'].item(), on_step=True, on_epoch=True, prog_bar=True,
                logger=False)
        self.log('loss_bc_cyc/train', loss_dict_cyc['loss_bc'].item(), on_step=True, on_epoch=True, prog_bar=True,
                logger=False)
        self.log('loss_objective_cyc/train', loss_dict_cyc['loss_objective'].item(), on_step=True, on_epoch=True, prog_bar=True,
                logger=False)
        self.logger.experiment.add_scalars('loss', {'loss_total': loss.item(),
                                                    'loss_box': loss_dict['loss_box'].item(),
                                                    'loss_seg': loss_dict['loss_seg'].item(),
                                                    'loss_vote': loss_dict['loss_vote'].item(),
                                                    'loss_objective': loss_dict['loss_objective'].item(),
                                                    'loss_bc': loss_dict['loss_bc'].item(),
                                                    'loss_box_cyc': loss_dict_cyc['loss_box'].item(),
                                                    'loss_seg_cyc': loss_dict_cyc['loss_seg'].item(),
                                                    'loss_vote_cyc': loss_dict_cyc['loss_vote'].item(),
                                                    'loss_objective_cyc': loss_dict_cyc['loss_objective'].item(),
                                                    'loss_bc_cyc': loss_dict_cyc['loss_bc'].item()},
                                        global_step=self.global_step)
            
                                           
        return {'loss':loss}
