from datasets import kitti, sampler, nuscenes_data, waymo_data


def get_dataset(config, type='train', cycle=False ,sample_rate = 1.0, **kwargs):
    if config.dataset == 'kitti':
        data = kitti.kittiDataset(path=config.path,
                                  split=kwargs.get('split', 'train'),
                                  category_name=config.category_name,
                                  coordinate_mode=config.coordinate_mode,
                                  preloading=config.preloading,
                                  preload_offset=config.preload_offset if type != 'test' else -1)
    elif config.dataset == 'nuscenes':
        data = nuscenes_data.NuScenesDataset(path=config.path,
                                             split=kwargs.get('split', 'train_track'),
                                             category_name=config.category_name,
                                             version=config.version,
                                             key_frame_only=True if type != 'test' else config.key_frame_only,
                                             # can only use keyframes for training
                                             preloading=config.preloading,
                                             preload_offset=config.preload_offset if type != 'test' else -1,
                                             min_points=1 if kwargs.get('split', 'train_track') in
                                                             [config.val_split, config.test_split] else -1)
    elif config.dataset == 'waymo':
        data = waymo_data.WaymoDataset(path=config.path,
                                       split=kwargs.get('split', 'train'),
                                       category_name=config.category_name,
                                       preloading=config.preloading,
                                       preload_offset=config.preload_offset,
                                       tiny=config.tiny)
    else:
        data = None

    if type == 'train_siamese':
        if cycle is True:
            return sampler.CycleTrackingSampler(dataset=data,
                                            random_sample=config.random_sample,
                                            sample_per_epoch=config.sample_per_epoch,
                                            sample_num = config.sample_num,
                                            sample_rate=config.sample_rate,
                                            seed=config.seed,
                                            config=config)
        if sample_rate != 1.0:
            return sampler.PointTrackingSampler(dataset=data,
                                                random_sample=config.random_sample,
                                                sample_per_epoch=config.sample_per_epoch,
                                                sample_rate=config.sample_rate,
                                                seed=config.seed,
                                                processing=sampler.semi_siamese_processing,
                                                config=config)
        return sampler.PointTrackingSampler(dataset=data,
                                                random_sample=config.random_sample,
                                                sample_per_epoch=config.sample_per_epoch,
                                                sample_rate=config.sample_rate,
                                                seed=config.seed,
                                                config=config)
        
    elif type.lower() == 'train_motion':
        return sampler.MotionTrackingSampler(dataset=data,
                                             config=config)
    else:
        return sampler.TestTrackingSampler(dataset=data, config=config)
