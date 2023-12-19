# Created by zenn at 2021/4/27
# Modified by qw at 2022/06/01

from copy import deepcopy
from turtle import backward
import numpy as np
from sklearn.preprocessing import scale
import torch
from easydict import EasyDict
from nuscenes.utils import geometry_utils

import datasets.points_utils as points_utils
from datasets.searchspace import KalmanFiltering


def no_processing(data, *args):
    return data


def siamese_processing(data, config, template_transform=None, search_transform=None):
    """

    :param data:
    :param config: {model_bb_scale,model_bb_offset,search_bb_scale, search_bb_offset}
    :return:
    """
    first_frame = data['first_frame']
    template_frame = data['template_frame']
    search_frame = data['search_frame']
    candidate_id = data['candidate_id']
    first_pc, first_box = first_frame['pc'], first_frame['3d_bbox']
    template_pc, template_box = template_frame['pc'], template_frame['3d_bbox']
    search_pc, search_box = search_frame['pc'], search_frame['3d_bbox']
    if template_transform is not None:
        template_pc, template_box = template_transform(template_pc, template_box)
        first_pc, first_box = template_transform(first_pc, first_box)
    if search_transform is not None:
        search_pc, search_box = search_transform(search_pc, search_box)
    # generating template. Merging the object from previous and the first frames.
    if candidate_id == 0:
        samplegt_offsets = np.zeros(3)
    else:
        samplegt_offsets = np.random.uniform(low=-0.3, high=0.3, size=3)
        samplegt_offsets[2] = samplegt_offsets[2] * (5 if config.degrees else np.deg2rad(5))
    template_box = points_utils.getOffsetBB(template_box, samplegt_offsets, limit_box=config.data_limit_box,
                                            degrees=config.degrees)
    model_pc, model_box = points_utils.getModel([first_pc, template_pc], [first_box, template_box],
                                                scale=config.model_bb_scale, offset=config.model_bb_offset)

    assert model_pc.nbr_points() > 20, 'not enough template points'

    # generating search area. Use the current gt box to select the nearby region as the search area.

    if candidate_id == 0 and config.num_candidates > 1:
        sample_offset = np.zeros(3)
    else:
        gaussian = KalmanFiltering(bnd=[1, 1, (5 if config.degrees else np.deg2rad(5))])
        sample_offset = gaussian.sample(1)[0]
    sample_bb = points_utils.getOffsetBB(search_box, sample_offset, limit_box=config.data_limit_box,
                                         degrees=config.degrees)
    search_pc_crop = points_utils.generate_subwindow(search_pc, sample_bb,
                                                     scale=config.search_bb_scale, offset=config.search_bb_offset)
    assert search_pc_crop.nbr_points() > 20, 'not enough search points'
    search_box = points_utils.transform_box(search_box, sample_bb)
    seg_label = points_utils.get_in_box_mask(search_pc_crop, search_box).astype(int)
    search_bbox_reg = [search_box.center[0], search_box.center[1], search_box.center[2], -sample_offset[2]]

    template_points, idx_t = points_utils.regularize_pc(model_pc.points.T, config.template_size)
    search_points, idx_s = points_utils.regularize_pc(search_pc_crop.points.T, config.search_size)
    seg_label = seg_label[idx_s]
    data_dict = {
        'template_points': template_points.astype('float32'),
        'search_points': search_points.astype('float32'),
        'box_label': np.array(search_bbox_reg).astype('float32'),
        'bbox_size': search_box.wlh,
        'seg_label': seg_label.astype('float32'),
        'mix_rate' : np.ones((1),dtype=np.float32)
    }
    if getattr(config, 'box_aware', False):
        template_bc = points_utils.get_point_to_box_distance(template_points, model_box)
        search_bc = points_utils.get_point_to_box_distance(search_points, search_box)
        data_dict.update({'points2cc_dist_t': template_bc.astype('float32'),
                          'points2cc_dist_s': search_bc.astype('float32'), })
    return data_dict

def semi_siamese_processing(data, config, template_transform=None, search_transform=None):
    """

    :param data:
    :param config: {model_bb_scale,model_bb_offset,search_bb_scale, search_bb_offset}
    :return:
    """
    search_frame = data['search_frame']
    mix1_frame = data['mixup_frame']
    candidate_id = data['candidate_id']
    mix_rate = data['mix_rate']
    mix1_pc, mix1_box = mix1_frame['pc'], mix1_frame['3d_bbox']
    search_pc, search_box = search_frame['pc'], search_frame['3d_bbox']
    if search_transform is not None:
        search_pc, search_box = search_transform(search_pc, search_box)
    # generating template. Merging the object from previous and the first frames.
    if candidate_id == 0:
        samplegt_offsets = np.zeros(3)
    else:
        samplegt_offsets = np.random.uniform(low=-0.3, high=0.3, size=3)
        samplegt_offsets[2] = samplegt_offsets[2] * (5 if config.degrees else np.deg2rad(5))
    model_box = points_utils.getOffsetBB(search_box, samplegt_offsets, limit_box=config.data_limit_box,
                                            degrees=config.degrees)
    model_pc, model_box = points_utils.getModel([search_pc], [model_box],
                                                scale=config.model_bb_scale, offset=config.model_bb_offset)

    assert model_pc.nbr_points() > 20, 'not enough template points'

    # generating search area. Use the current gt box to select the nearby region as the search area.

    if candidate_id == 0 and config.num_candidates > 1:
        sample_offset = np.zeros(3)
    else:
        gaussian = KalmanFiltering(bnd=[1, 1, (5 if config.degrees else np.deg2rad(5))])
        sample_offset = gaussian.sample(1)[0]

    sample_bb = points_utils.getOffsetBB(search_box, sample_offset, limit_box=config.data_limit_box,
                                         degrees=config.degrees)
    search_pc_crop = points_utils.generate_subwindow(search_pc, sample_bb,
                                                     scale=config.search_bb_scale, offset=config.search_bb_offset)
    assert search_pc_crop.nbr_points() > 20, 'not enough search points'
    search_box = points_utils.transform_box(search_box, sample_bb)
    seg_label = points_utils.get_in_box_mask(search_pc_crop, search_box).astype(int)
    search_bbox_reg = [search_box.center[0], search_box.center[1], search_box.center[2], -sample_offset[2]]

    template_points, idx_t = points_utils.regularize_pc(model_pc.points.T, config.template_size)
    search_points, idx_s = points_utils.regularize_pc(search_pc_crop.points.T, config.search_size)
    seg_label = seg_label[idx_s]
    data_dict = {
        'template_points': template_points.astype('float32'),
        'search_points': search_points.astype('float32'),
        'box_label': np.array(search_bbox_reg).astype('float32'),
        'bbox_size': search_box.wlh,
        'seg_label': seg_label.astype('float32'),
        'mix_rate': mix_rate.astype('float32'),
    }
    if getattr(config, 'box_aware', False):
        template_bc = points_utils.get_point_to_box_distance(template_points, model_box)
        search_bc = points_utils.get_point_to_box_distance(search_points, search_box)
        data_dict.update({'points2cc_dist_t': template_bc.astype('float32'),
                          'points2cc_dist_s': search_bc.astype('float32'), })
    return data_dict


def motion_processing(data, config, template_transform=None, search_transform=None):
    """

    :param data:
    :param config: {model_bb_scale,model_bb_offset,search_bb_scale, search_bb_offset}
    :return:
    point_sample_size
    bb_scale
    bb_offset
    """
    prev_frame = data['prev_frame']
    this_frame = data['this_frame']
    candidate_id = data['candidate_id']
    prev_pc, prev_box = prev_frame['pc'], prev_frame['3d_bbox']
    this_pc, this_box = this_frame['pc'], this_frame['3d_bbox']

    num_points_in_prev_box = geometry_utils.points_in_box(prev_box, prev_pc.points).sum()
    assert num_points_in_prev_box > 10, 'not enough target points'

    if template_transform is not None:
        prev_pc, prev_box = template_transform(prev_pc, prev_box)
    if search_transform is not None:
        this_pc, this_box = search_transform(this_pc, this_box)

    if candidate_id == 0:
        sample_offsets = np.zeros(3)
    else:
        sample_offsets = np.random.uniform(low=-0.3, high=0.3, size=3)
        sample_offsets[2] = sample_offsets[2] * (5 if config.degrees else np.deg2rad(5))
    ref_box = points_utils.getOffsetBB(prev_box, sample_offsets, limit_box=config.data_limit_box,
                                       degrees=config.degrees)
    prev_frame_pc = points_utils.generate_subwindow(prev_pc, ref_box,
                                                    scale=config.bb_scale,
                                                    offset=config.bb_offset)

    this_frame_pc = points_utils.generate_subwindow(this_pc, ref_box,
                                                    scale=config.bb_scale,
                                                    offset=config.bb_offset)
    assert this_frame_pc.nbr_points() > 20, 'not enough search points'

    this_box = points_utils.transform_box(this_box, ref_box)
    prev_box = points_utils.transform_box(prev_box, ref_box)
    ref_box = points_utils.transform_box(ref_box, ref_box)
    motion_box = points_utils.transform_box(this_box, prev_box)

    prev_points, idx_prev = points_utils.regularize_pc(prev_frame_pc.points.T, config.point_sample_size)
    this_points, idx_this = points_utils.regularize_pc(this_frame_pc.points.T, config.point_sample_size)

    seg_label_this = geometry_utils.points_in_box(this_box, this_points.T, 1.25).astype(int)
    seg_label_prev = geometry_utils.points_in_box(prev_box, prev_points.T, 1.25).astype(int)
    seg_mask_prev = geometry_utils.points_in_box(ref_box, prev_points.T, 1.25).astype(float)
    if candidate_id != 0:
        # Here we use 0.2/0.8 instead of 0/1 to indicate that the previous box is not GT.
        # When boxcloud is used, the actual value of prior-targetness mask doesn't really matter.
        seg_mask_prev[seg_mask_prev == 0] = 0.2
        seg_mask_prev[seg_mask_prev == 1] = 0.8
    seg_mask_this = np.full(seg_mask_prev.shape, fill_value=0.5)

    timestamp_prev = np.full((config.point_sample_size, 1), fill_value=0)
    timestamp_this = np.full((config.point_sample_size, 1), fill_value=0.1)

    prev_points = np.concatenate([prev_points, timestamp_prev, seg_mask_prev[:, None]], axis=-1)
    this_points = np.concatenate([this_points, timestamp_this, seg_mask_this[:, None]], axis=-1)

    stack_points = np.concatenate([prev_points, this_points], axis=0)
    stack_seg_label = np.hstack([seg_label_prev, seg_label_this])
    theta_this = this_box.orientation.degrees * this_box.orientation.axis[-1] if config.degrees else \
        this_box.orientation.radians * this_box.orientation.axis[-1]
    box_label = np.append(this_box.center, theta_this).astype('float32')
    theta_prev = prev_box.orientation.degrees * prev_box.orientation.axis[-1] if config.degrees else \
        prev_box.orientation.radians * prev_box.orientation.axis[-1]
    box_label_prev = np.append(prev_box.center, theta_prev).astype('float32')
    theta_motion = motion_box.orientation.degrees * motion_box.orientation.axis[-1] if config.degrees else \
        motion_box.orientation.radians * motion_box.orientation.axis[-1]
    motion_label = np.append(motion_box.center, theta_motion).astype('float32')

    motion_state_label = np.sqrt(np.sum((this_box.center - prev_box.center) ** 2)) > config.motion_threshold

    data_dict = {
        'points': stack_points.astype('float32'),
        'box_label': box_label,
        'box_label_prev': box_label_prev,
        'motion_label': motion_label,
        'motion_state_label': motion_state_label.astype('int'),
        'bbox_size': this_box.wlh,
        'seg_label': stack_seg_label.astype('int'),
    }

    if getattr(config, 'box_aware', False):
        prev_bc = points_utils.get_point_to_box_distance(stack_points[:config.point_sample_size, :3], prev_box)
        this_bc = points_utils.get_point_to_box_distance(stack_points[config.point_sample_size:, :3], this_box)
        candidate_bc_prev = points_utils.get_point_to_box_distance(stack_points[:config.point_sample_size, :3], ref_box)
        candidate_bc_this = np.zeros_like(candidate_bc_prev)
        candidate_bc = np.concatenate([candidate_bc_prev, candidate_bc_this], axis=0)

        data_dict.update({'prev_bc': prev_bc.astype('float32'),
                          'this_bc': this_bc.astype('float32'),
                          'candidate_bc': candidate_bc.astype('float32')})
    return data_dict


def Cycle_processing(data, config):
    """

    :param data:
    :param config: {model_bb_scale,model_bb_offset,search_bb_scale, search_bb_offset}
    :return:
    """
    first_frame = data['first_frame']
    template_frame = data['template_frame']
    second_frame = data['second_frame']
    candidate_id = data['candidate_id']
    mix1_frame = data['mixup_frame1']
    mix2_frame = data['mixup_frame2']
    first_pc, first_box = first_frame['pc'], first_frame['3d_bbox']
    template_pc, template_box = template_frame['pc'], template_frame['3d_bbox']
    second_pc, second_box = second_frame['pc'], second_frame['3d_bbox']
    mix1_pc, mix1_box = mix1_frame['pc'], mix1_frame['3d_bbox']
    mix2_pc, mix2_box = mix2_frame['pc'], mix2_frame['3d_bbox']
    mix_rate = data['mix_rate']

    # generating template. Merging the object from previous and the first frames.
    if candidate_id == 0:
        samplegt_offsets = np.zeros(3)
    else:
        samplegt_offsets = np.random.uniform(low=-0.3, high=0.3, size=3)
        samplegt_offsets[2] = samplegt_offsets[2] * (5 if config.degrees else np.deg2rad(5))
    model_box = points_utils.getOffsetBB(template_box, samplegt_offsets, limit_box=config.limit_box, degrees=config.degrees)

    model_pc, model_box = points_utils.getModel([template_pc], [model_box],
                                                scale=config.model_bb_scale, offset=config.model_bb_offset)
        
    assert model_pc.nbr_points() > 20, 'not enough template points'

    # generating search area. Use the current gt box to select the nearby region as the search area.
    if candidate_id == 0 and config.num_candidates > 1:
        sample_offset = np.zeros(3)
        mix_rate = np.ones((2),dtype=np.float32)
    else:
        gaussian = KalmanFiltering(bnd=[1, 1, (5 if config.degrees else np.deg2rad(5))])
        sample_offset = gaussian.sample(1)[0]
    if not (config.mix_up_search or config.mix_up_template):
        mix_rate = np.ones((2),dtype=np.float32)
    #contral mix up in each frame !!!!! be careful !!!!!!!
    mix_rate[1] = 1.0
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    #mix_up search area
    if config.mix_up_search:
        mix1_template_pc,mix1_template_box,mix_rate[0] = points_utils.generate_mixpc(template_pc,mix1_pc,template_box,mix1_box,
                                            mix_rate[0],offset=config.forward_bb_offset,scale=config.forward_bb_scale,seed=1)
        mix2_template_pc,mix2_template_box,mix_rate[1] = points_utils.generate_mixpc(template_pc,mix2_pc,template_box,mix2_box,
                                            mix_rate[1],offset=config.forward_bb_offset,scale=config.forward_bb_scale,seed=1)
    else:
        mix1_template_pc = template_pc
        mix2_template_pc = template_pc
        mix1_template_box = template_box
        mix2_template_box = template_box
    #mix_up template
    if config.mix_up_template:
        model_pc,model_box,mix_rate[0] = points_utils.generate_mixpc_template(model_pc,mix1_pc,model_box,mix1_box,
                                            mix_rate[0],seed=1)
        # print(len(sample_offset))
    if config.mix_up_search and config.mix_up_template:
        mix_rate[0] = 1.0
    backward_template_frame = deepcopy(template_frame)
    backward_template_frame['pc'] = mix2_template_pc
    backward_template_frame['3d_bbox'] = template_box
    sample_bb = points_utils.getOffsetBB(template_box, sample_offset, limit_box=config.limit_box, degrees=config.degrees)
    search_pc_crop = points_utils.generate_subwindow(mix1_template_pc, sample_bb,
                                                     scale=config.search_bb_scale, offset=config.search_bb_offset)
    assert search_pc_crop.nbr_points() > 20, 'not enough search points'
    if config.transformation:
        backward_offset = np.random.uniform(low=-0.3, high=0.3, size=(2,3))
        backward_offset[:,2] = backward_offset[:,2] * (5 if config.degrees else np.deg2rad(5))
    else:
        backward_offset = np.zeros((2,3))
    # backward_sample_bb = points_utils.getOffsetBB(sample_bb, backward_offset[0], limit_box=config.limit_box, degrees=config.degrees)
    # backward_template_pc = points_utils.crop_pc_axis_aligned(template_pc, backward_sample_bb,
    #                                                  scale=config.forward_bb_scale, offset=config.forward_bb_offset)
    # backward_template_frame = deepcopy(template_frame)
    # backward_template_frame['pc'] = backward_template_pc

    gaussian = KalmanFiltering(bnd=[1, 1, (5 if config.degrees else np.deg2rad(5))])
    forward_offset = gaussian.sample(2)
    forward_box = points_utils.getOffsetBB(template_box,forward_offset[0],limit_box=config.limit_box, degrees=config.degrees)
    forward_pc1 = points_utils.crop_pc_axis_aligned(first_pc, forward_box,
                                                    scale=config.forward_bb_scale, offset=config.forward_bb_offset)
    # backward_box = points_utils.getOffsetBB(forward_box, backward_offset[1], limit_box=config.limit_box, degrees=config.degrees)
    # backward_pc1 = points_utils.crop_pc_axis_aligned(first_pc, backward_box,
    #                                                 scale=config.forward_bb_scale, offset=config.forward_bb_offset)
    first_frame['pc'] = forward_pc1
    # backward_frame1 = deepcopy(first_frame)
    # backward_frame1['pc'] = backward_pc1

    forward_box = points_utils.getOffsetBB(forward_box, forward_offset[1], limit_box=config.limit_box, degrees=config.degrees)
    forward_pc2 = points_utils.crop_pc_axis_aligned(second_pc, forward_box,
                                                    scale=config.forward_bb_scale, offset=config.forward_bb_offset)
    second_frame['pc'] = forward_pc2
    template_box = points_utils.transform_box(template_box, sample_bb)
    seg_label = points_utils.get_in_box_mask(search_pc_crop, template_box).astype(int)
    search_bbox_reg = [template_box.center[0], template_box.center[1], template_box.center[2], -sample_offset[2]]

    template_points, idx_t = points_utils.regularize_pc(model_pc.points.T, config.template_size)
    search_points, idx_s = points_utils.regularize_pc(search_pc_crop.points.T, config.search_size)
    seg_label = seg_label[idx_s]
    data_dict = {
        'template_points': template_points.astype('float32'),
        'search_points': search_points.astype('float32'),
        'box_label': np.array(search_bbox_reg).astype('float32'),
        'bbox_size': template_box.wlh,
        'seg_label': seg_label.astype('float32'),
        'template_frame': template_frame,
        'forward1_frame': first_frame,
        'forward2_frame': second_frame,
        'backward_frame1':first_frame,
        'backward_template_frame': backward_template_frame,
        # 'backward_frame1':backward_frame1,
        # 'backward_template_frame': backward_template_frame,
        'backward_offset': np.array(backward_offset).astype('float32'),
        'mix_rate': mix_rate.astype('float32')
    }
    if getattr(config, 'box_aware', False):
        template_bc = points_utils.get_point_to_box_distance(template_points, model_box)
        search_bc = points_utils.get_point_to_box_distance(search_points, template_box)
        data_dict.update({'points2cc_dist_t': template_bc.astype('float32'),
                          'points2cc_dist_s': search_bc.astype('float32'), })
    return data_dict

class CycleTrackingSampler(torch.utils.data.Dataset):
    #@param: sample_num: number of frames in cycle ;!!!!!!Only support 3 now !!!!!!!!!!!!!!!!
    def __init__(self, dataset, sample_rate=1.0, sample_num=3,seed = 1,random_sample = False, sample_per_epoch=10000, processing=Cycle_processing,config=None,
                 **kwargs):
        if config is None:
            config = EasyDict(kwargs)
        self.sample_per_epoch = sample_per_epoch
        self.dataset = dataset
        self.processing = processing
        self.config = config
        self.sample_rate = sample_rate
        self.sample_num = sample_num
        self.random_sample = random_sample
        self.generator = np.random.default_rng(seed)
        self.sample_list =  self.generator.choice(self.dataset.get_num_frames_total(),size=int(self.dataset.get_num_frames_total() * self.sample_rate),replace=False)
        self.num_candidates = getattr(config, 'num_candidates', 1)
        if not self.random_sample:
            num_frames_total = 0
            self.tracklet_start_ids = [num_frames_total]
            for i in range(dataset.get_num_tracklets()):
                num_frames_total += dataset.get_num_frames_tracklet(i)
                self.tracklet_start_ids.append(num_frames_total)

    def get_anno_index(self, index):
        return self.sample_list[index // self.num_candidates]

    def get_candidate_index(self, index):
        return index % self.num_candidates

    def __len__(self):
        if self.random_sample:
            return self.sample_per_epoch * self.num_candidates
        else:
            return int(self.dataset.get_num_frames_total() * self.sample_rate)  * self.num_candidates 

    def __getitem__(self, index):
        anno_id = self.get_anno_index(index)
        candidate_id = self.get_candidate_index(index)
        mixup_id = self.generator.choice(self.sample_list,size=2)
        try:
            if self.random_sample:
                tracklet_id = torch.randint(0, self.dataset.get_num_tracklets(), size=(1,)).item()
                tracklet_annos = self.dataset.tracklet_anno_list[tracklet_id]
                frame_ids = [0] + points_utils.random_choice(num_samples=2, size=len(tracklet_annos)).tolist()
            else:
                frame_ids = None
                for i in range(0, self.dataset.get_num_tracklets()):
                    if self.tracklet_start_ids[i] <= anno_id < self.tracklet_start_ids[i + 1]:
                        tracklet_id = i
                        this_frame_id = anno_id - self.tracklet_start_ids[i]
                        frame_ids = np.arange(self.sample_num, dtype=np.int64)
                        if self.tracklet_start_ids[i+1] - anno_id < self.sample_num:
                            frame_ids[self.tracklet_start_ids[i+1] - anno_id:] = self.tracklet_start_ids[i+1] - anno_id - 1
                        frame_ids += this_frame_id
                    if self.tracklet_start_ids[i] <= mixup_id[0] < self.tracklet_start_ids[i + 1]:
                        mix1_tracklet_id = i
                        mix1_frame_id = mixup_id[0] - self.tracklet_start_ids[i]
                    if self.tracklet_start_ids[i] <= mixup_id[1] < self.tracklet_start_ids[i + 1]:
                        mix2_tracklet_id = i
                        mix2_frame_id = mixup_id[1] - self.tracklet_start_ids[i]
                    
            template_frame, first_frame, second_frame = self.dataset.get_frames(tracklet_id, frame_ids=frame_ids)
            mixup_frame1 = self.dataset.get_frames(mix1_tracklet_id, frame_ids=[mix1_frame_id])[0]
            mixup_frame2 = self.dataset.get_frames(mix2_tracklet_id, frame_ids=[mix2_frame_id])[0]
            mix_rate = self.generator.beta(0.5,0.5,(2))
            for idx in range(2):
                if mixup_id[idx] == anno_id:
                        mix_rate[idx] = 1.0
            # print(mix_rate.shape)
            data = {"first_frame": first_frame,
                    "template_frame": template_frame,
                    "second_frame": second_frame,
                    "candidate_id": candidate_id,
                    "mixup_frame1":mixup_frame1,
                    "mixup_frame2":mixup_frame2,
                    "mix_rate":mix_rate,}

            return self.processing(data, self.config)
        except AssertionError:
            return self[torch.randint(0, len(self), size=(1,)).item()]



class PointTrackingSampler(torch.utils.data.Dataset):
    def __init__(self, dataset, random_sample, sample_rate=1.0 ,seed = 1, sample_per_epoch=10000, processing=siamese_processing, config=None,
                 **kwargs):
        if config is None:
            config = EasyDict(kwargs)
        self.sample_per_epoch = sample_per_epoch
        self.dataset = dataset
        self.processing = processing
        self.config = config
        self.sample_rate = sample_rate
        self.random_sample = random_sample
        self.generator = np.random.default_rng(seed)
        self.sample_list =  self.generator.choice(self.dataset.get_num_frames_total(),size=int(self.dataset.get_num_frames_total() * self.sample_rate),replace=False)
        self.num_candidates = getattr(config, 'num_candidates', 1)
        if getattr(self.config, "use_augmentation", False):
            print('using augmentation')
            self.transform = points_utils.apply_augmentation
        else:
            self.transform = None
        if not self.random_sample:
            num_frames_total = 0
            self.tracklet_start_ids = [num_frames_total]
            for i in range(dataset.get_num_tracklets()):
                num_frames_total += dataset.get_num_frames_tracklet(i)
                self.tracklet_start_ids.append(num_frames_total)

    def get_anno_index(self, index):
        return self.sample_list[index // self.num_candidates]

    def get_candidate_index(self, index):
        return index % self.num_candidates

    def __len__(self):
        if self.random_sample:
            return self.sample_per_epoch * self.num_candidates
        else:
            return int(self.dataset.get_num_frames_total() * self.sample_rate)* self.num_candidates
    def __getitem__(self, index):
        anno_id = self.get_anno_index(index)
        candidate_id = self.get_candidate_index(index)
        mixup_id = self.generator.choice(self.sample_list,size=2)
        try:
            if self.random_sample:
                tracklet_id = torch.randint(0, self.dataset.get_num_tracklets(), size=(1,)).item()
                tracklet_annos = self.dataset.tracklet_anno_list[tracklet_id]
                frame_ids = [0] + points_utils.random_choice(num_samples=2, size=len(tracklet_annos)).tolist()
            else:
                for i in range(0, self.dataset.get_num_tracklets()):
                    if self.tracklet_start_ids[i] <= anno_id < self.tracklet_start_ids[i + 1]:
                        tracklet_id = i
                        this_frame_id = anno_id - self.tracklet_start_ids[i]
                        prev_frame_id = max(this_frame_id - 1, 0)
                        frame_ids = (0, prev_frame_id, this_frame_id)
                    if self.tracklet_start_ids[i] <= mixup_id[0] < self.tracklet_start_ids[i + 1]:
                        mix1_tracklet_id = i
                        mix1_frame_id = mixup_id[0] - self.tracklet_start_ids[i]
            first_frame, template_frame, search_frame = self.dataset.get_frames(tracklet_id, frame_ids=frame_ids)
            mixup_frame = self.dataset.get_frames(mix1_tracklet_id, frame_ids=[mix1_frame_id])[0]
            mix_rate = self.generator.beta(0.5,0.5,(1))
            if mixup_id[0] == anno_id:
                mix_rate[0] = 1.0
            data = {"first_frame": first_frame,
                    "template_frame": template_frame,
                    "search_frame": search_frame,
                    "candidate_id": candidate_id,
                    "mixup_frame": mixup_frame,
                    "mix_rate": mix_rate,}

            return self.processing(data, self.config,
                                   template_transform=None,
                                   search_transform=self.transform)
        except AssertionError:
            return self[torch.randint(0, len(self), size=(1,)).item()]


class TestTrackingSampler(torch.utils.data.Dataset):
    def __init__(self, dataset, config=None, **kwargs):
        if config is None:
            config = EasyDict(kwargs)
        self.dataset = dataset
        self.config = config

    def __len__(self):
        return self.dataset.get_num_tracklets()

    def __getitem__(self, index):
        tracklet_annos = self.dataset.tracklet_anno_list[index]
        frame_ids = list(range(len(tracklet_annos)))
        return self.dataset.get_frames(index, frame_ids)


class MotionTrackingSampler(PointTrackingSampler):
    def __init__(self, dataset, config=None, **kwargs):
        super().__init__(dataset, random_sample=False, config=config, **kwargs)
        self.processing = motion_processing

    def __getitem__(self, index):
        anno_id = self.get_anno_index(index)
        candidate_id = self.get_candidate_index(index)
        try:

            for i in range(0, self.dataset.get_num_tracklets()):
                if self.tracklet_start_ids[i] <= anno_id < self.tracklet_start_ids[i + 1]:
                    tracklet_id = i
                    this_frame_id = anno_id - self.tracklet_start_ids[i]
                    prev_frame_id = max(this_frame_id - 1, 0)
                    frame_ids = (0, prev_frame_id, this_frame_id)
            first_frame, prev_frame, this_frame = self.dataset.get_frames(tracklet_id, frame_ids=frame_ids)
            data = {
                "first_frame": first_frame,
                "prev_frame": prev_frame,
                "this_frame": this_frame,
                "candidate_id": candidate_id}
            return self.processing(data, self.config,
                                   template_transform=self.transform,
                                   search_transform=self.transform)
        except AssertionError:
            return self[torch.randint(0, len(self), size=(1,)).item()]
