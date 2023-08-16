"""
main.py
Created by zenn at 2021/7/18 15:08
"""
import pytorch_lightning as pl
import argparse

import pytorch_lightning.utilities.distributed
import torch
import yaml
from easydict import EasyDict
import os

from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from torch.utils.data import DataLoader

from datasets import get_dataset
from models import get_model

import numpy as np

# os.environ["NCCL_DEBUG"] = "INFO"

def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=48, help='input batch size')
    parser.add_argument('--epoch', type=int, default=200, help='number of epochs')
    parser.add_argument('--save_top_k', type=int, default=-1, help='save top k checkpoints')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='check_val_every_n_epoch')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--cfg', type=str, help='the config_file')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint location')
    parser.add_argument('--resume', action='store_true', default=False, help='resume checkpoint')
    parser.add_argument('--log_dir', type=str, default=None, help='log location')
    parser.add_argument('--test', action='store_true', default=False, help='test mode')
    parser.add_argument('--preloading', action='store_true', default=False, help='preload dataset into memory')

    args = parser.parse_args()
    config = load_yaml(args.cfg)
    config.update(vars(args))  # override the configuration using the value in args

    return EasyDict(config)


cfg = parse_config()
env_cp = os.environ.copy()


def clooate_fn(batch):
    template_points,search_points,box_label,bbox_size,seg_label,template_frame,forward1_frame,\
    forward2_frame,backward_frame1,backward_template_frame,backward_offset, mix_rate = [],[],[],[],[],[],[],[],[],[],[],[]
    for data_dict in batch:
        template_points += [data_dict['template_points']]
        search_points += [data_dict['search_points']]
        box_label += [data_dict['box_label']]
        bbox_size += [data_dict['bbox_size']]
        seg_label += [data_dict['seg_label']]
        template_frame += [data_dict['template_frame']]
        forward1_frame += [data_dict['forward1_frame']]
        forward2_frame += [data_dict['forward2_frame']]
        backward_frame1 += [data_dict['backward_frame1']]
        backward_template_frame += [data_dict['backward_template_frame']]
        backward_offset += [data_dict['backward_offset']]
        mix_rate += [data_dict['mix_rate']]

    template_points = np.stack(template_points)
    search_points = np.stack(search_points)
    box_label = np.stack(box_label)
    bbox_size = np.stack(bbox_size)
    seg_label = np.stack(seg_label)
    backward_offset = np.stack(backward_offset)
    mix_rate = np.stack(mix_rate)

    template_points = torch.from_numpy(template_points)
    search_points = torch.from_numpy(search_points)
    box_label = torch.from_numpy(box_label)
    bbox_size = torch.from_numpy(bbox_size)
    seg_label = torch.from_numpy(seg_label)
    mix_rate = torch.from_numpy(mix_rate)
    
    dict = {
        'template_points': template_points,
        'search_points': search_points,
        'box_label': box_label,
        'bbox_size': bbox_size,
        'seg_label': seg_label,
        'template_frame': template_frame,
        'forward1_frame': forward1_frame,
        'forward2_frame': forward2_frame,
        'backward_frame1': backward_frame1,
        'backward_template_frame': backward_template_frame,
        'backward_offset': backward_offset,
        'mix_rate': mix_rate
    }
    if getattr(cfg, 'box_aware', False):
        points2cc_dist_t,points2cc_dist_s = [],[]
        for data_dict in batch:
            points2cc_dist_t += [data_dict['points2cc_dist_t']]
            points2cc_dist_s += [data_dict['points2cc_dist_s']]
        points2cc_dist_t = np.stack(points2cc_dist_t)
        points2cc_dist_s = np.stack(points2cc_dist_s)
        points2cc_dist_t = torch.from_numpy(points2cc_dist_t)
        points2cc_dist_s = torch.from_numpy(points2cc_dist_s)
        
        dict.update({'points2cc_dist_t': points2cc_dist_t,
                    'points2cc_dist_s': points2cc_dist_s, })
    
    return dict


try:
    node_rank, local_rank, world_size = env_cp['NODE_RANK'], env_cp['LOCAL_RANK'], env_cp['WORLD_SIZE']

    is_in_ddp_subprocess = env_cp['PL_IN_DDP_SUBPROCESS']
    pl_trainer_gpus = env_cp['PL_TRAINER_GPUS']
    print(node_rank, local_rank, world_size, is_in_ddp_subprocess, pl_trainer_gpus)

    if int(local_rank) == int(world_size) - 1:
        print(cfg)
except KeyError:
    pass

# init model
if cfg.checkpoint is None:
    net = get_model(cfg.net_model)(cfg)
else:
    net = get_model(cfg.net_model).load_from_checkpoint(cfg.checkpoint, config=cfg)
if not cfg.test:
    # dataset and dataloader
    train_data = get_dataset(cfg, type=cfg.train_type, split=cfg.train_split,cycle=cfg.cycle,sample_rate=cfg.sample_rate)
    val_data = get_dataset(cfg, type='test', split=cfg.val_split)
    print(len(train_data))
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, num_workers=cfg.workers, shuffle=True,drop_last=True,
                              pin_memory=True,collate_fn=clooate_fn if cfg.cycle else None)
    val_loader = DataLoader(val_data, batch_size=1, num_workers=cfg.workers, collate_fn=lambda x: x, pin_memory=True)
    checkpoint_callback = ModelCheckpoint(monitor='precision/test', mode='max', save_last=True,
                                          save_top_k=cfg.save_top_k)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # init trainer
    trainer = pl.Trainer(gpus=-1, accelerator='ddp', max_epochs=cfg.epoch, resume_from_checkpoint=cfg.checkpoint if cfg.resume else None,
                         callbacks=[checkpoint_callback,lr_monitor], default_root_dir=cfg.log_dir,
                         check_val_every_n_epoch=cfg.check_val_every_n_epoch, num_sanity_val_steps=2,
                         gradient_clip_val=cfg.gradient_clip_val,auto_lr_find= False)
    trainer.fit(net, train_loader, val_loader)
else:
    test_data = get_dataset(cfg, type='test', split=cfg.test_split)
    test_loader = DataLoader(test_data, batch_size=1, num_workers=cfg.workers, collate_fn=lambda x: x, pin_memory=True)

    trainer = pl.Trainer(gpus=-1, accelerator='ddp', default_root_dir=cfg.log_dir,
                         resume_from_checkpoint=cfg.checkpoint)
    trainer.test(net, test_loader)
