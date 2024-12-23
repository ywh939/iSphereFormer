import os
import random
import glob
import numpy as np
import torch
import yaml
import pickle
from util.data_util import data_prepare
from pathlib import Path


class SemanticCustom(torch.utils.data.Dataset):
    def __init__(self, 
        data_path, 
        voxel_size=[0.1, 0.1, 0.1], 
        split='train', 
        return_ref=True, 
        label_mapping="", 
        rotate_aug=True, 
        flip_aug=True, 
        scale_aug=True, 
        scale_params=[0.95, 1.05], 
        transform_aug=True, 
        trans_std=[0.1, 0.1, 0.1],
        ignore_label=255, 
        voxel_max=None, 
        xyz_norm=False, 
        pc_range=None, 
        use_tta=None,
        vote_num=4,
        tempo_sample_num=1
    ):
        super().__init__()
        self.num_classes = 19
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.return_ref = return_ref
        self.split = split
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.scale_params = scale_params
        self.transform_aug = transform_aug
        self.trans_std = trans_std
        self.ignore_label = ignore_label
        self.voxel_max = voxel_max
        self.xyz_norm = xyz_norm
        self.pc_range = None if pc_range is None else np.array(pc_range)
        self.data_path = data_path
        self.use_tta = use_tta
        self.vote_num = vote_num

        self.tempo_sample_num = tempo_sample_num

        if split == 'train':
            splits = semkittiyaml['split']['valid']
        elif split == 'val':
            splits = semkittiyaml['split']['valid']
        elif split == 'test':
            splits = semkittiyaml['split']['test']
        elif split == 'trainval':
            splits = semkittiyaml['split']['train'] + semkittiyaml['split']['valid']
        else:
            raise Exception('Split must be train/val/test')
        
        self.files = []
        
        for i_folder in splits:
            self.files += sorted(glob.glob(os.path.join(data_path, i_folder, 'kitti_format', "*.bin")))

        if isinstance(voxel_size, list):
            voxel_size = np.array(voxel_size).astype(np.float32)
        self.voxel_size = voxel_size

    def __len__(self):
        'Denotes the total number of samples'
        # return len(self.nusc_infos)
        return len(self.files)

    def __getitem__(self, index):
        if self.use_tta:
            samples = []
            for i in range(self.vote_num):
                sample = tuple(self.get_single_sample(index, vote_idx=i))
                samples.append(sample)
            return tuple(samples)
        return self.get_single_sample(index)
    
    def get_tempo_samples(self, index, cur_coords, cur_xyz, cur_feats, cur_labels):
        cur_sample_info = {'index' : index, 
                       'coords' : cur_coords, 
                       'xyz' : cur_xyz, 
                       'feats' : cur_feats, 
                       'labels' : cur_labels}
        
        tempo_feats = [cur_sample_info]

        for idx in range(index - 1, index - self.tempo_sample_num, -1):
            if (idx < 0):
                tempo_feats.append(tempo_feats[-1])
                continue

            file_path = self.files[idx]
            raw_data = np.fromfile(file_path, dtype=np.float32).reshape((-1, 3))
            p = np.ones(raw_data.shape[0]).reshape(-1, 1)
            points = np.hstack((raw_data, p))
            raw_data = points

            annotated_data = np.fromfile(file_path[:-3] + 'label', dtype=np.uint32).reshape((-1, 1))
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)
            annotated_data = annotated_data - 1
            labels_in = annotated_data.astype(np.uint8).reshape(-1)

            sample_info = { 'index': idx }
            if self.split == 'train':
                t_coords, t_xyz, t_feats, t_labels = data_prepare(raw_data[:, :3], raw_data[:, :4], labels_in, self.split, self.voxel_size, self.voxel_max, None, self.xyz_norm)
            else:
                t_coords, t_xyz, t_feats, t_labels, _ = data_prepare(raw_data[:, :3], raw_data[:, :4], labels_in, self.split, self.voxel_size, self.voxel_max, None, self.xyz_norm)
            sample_info['coords'] = t_coords
            sample_info['xyz'] = t_xyz
            sample_info['feats'] = t_feats
            sample_info['labels'] = t_labels
            tempo_feats.append(sample_info)

        return tempo_feats

    def get_single_sample(self, index, vote_idx=0):

        file_path = self.files[index]

        raw_data = np.fromfile(file_path, dtype=np.float32).reshape((-1, 3))
        annotated_data = np.fromfile(file_path[:-3] + 'label', dtype=np.uint32).reshape((-1, 1))
        annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        # points = raw_data
        p = np.ones(raw_data.shape[0]).reshape(-1, 1)
        points = np.hstack((raw_data, p))

        if self.split != 'test':
            # annotated_data[annotated_data == 0] = self.ignore_label + 1
            annotated_data = annotated_data - 1
            labels_in = annotated_data.astype(np.uint8).reshape(-1)
        else:
            labels_in = np.zeros(points.shape[0]).astype(np.uint8)

        # Augmentation
        # ==================================================
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360) - np.pi
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            points[:, :2] = np.dot(points[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            if self.use_tta:
                flip_type = vote_idx % 4
            else:
                flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                points[:, 0] = -points[:, 0]
            elif flip_type == 2:
                points[:, 1] = -points[:, 1]
            elif flip_type == 3:
                points[:, :2] = -points[:, :2]

        if self.scale_aug:
            noise_scale = np.random.uniform(self.scale_params[0], self.scale_params[1])
            points[:, 0] = noise_scale * points[:, 0]
            points[:, 1] = noise_scale * points[:, 1]
            
        if self.transform_aug:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T
            points[:, 0:3] += noise_translate
            
        # ==================================================

        feats = points
        xyz = points[:, :3]

        if self.pc_range is not None:
            xyz = np.clip(xyz, self.pc_range[0], self.pc_range[1])

        if self.split == 'train':
            coords, xyz, feats, labels = data_prepare(xyz, feats, labels_in, self.split, self.voxel_size, self.voxel_max, None, self.xyz_norm)

            if self.tempo_sample_num > 1:
                tempo_data = self.get_tempo_samples(index, coords, xyz, feats, labels)
                return coords, xyz, feats, labels, tempo_data
            else:
                return coords, xyz, feats, labels
        else:
            coords, xyz, feats, labels, inds_reconstruct = data_prepare(xyz, feats, labels_in, self.split, self.voxel_size, self.voxel_max, None, self.xyz_norm)

            filepath = Path(file_path)
            filename = filepath.stem
            filep = str(filepath.parent)

            if self.split == 'val':
                if self.tempo_sample_num > 1:
                    tempo_data = self.get_tempo_samples(index, coords, xyz, feats, labels)
                    return coords, xyz, feats, labels, inds_reconstruct, tempo_data
                else:
                    return coords, xyz, feats, labels, inds_reconstruct, filename
            elif self.split == 'test':
                # return coords, xyz, feats, labels, inds_reconstruct, filename, self.files[index]
                return coords, xyz, feats, labels, inds_reconstruct, filename
