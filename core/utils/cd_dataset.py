import os
import cv2
import pdb
import numpy as np
import random
from skimage import io
import torch
from torch.utils.data import Dataset
from datasets import datasets_catalog as dc
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import (
    Compose,
    RandomBrightnessContrast,
    HueSaturationValue,
    RGBShift
)
import core.utils.imutils as imutils


# from ever by Zhuo Zheng
class ToTensor(ToTensorV2):
    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask, 'masks': self.apply_to_masks}

    def apply_to_masks(self, masks, **params):
        return [self.apply_to_mask(m, **params) for m in masks]


class BDADataset_infer(Dataset):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.datasets_name = cfg.TRAIN.DATASETS
        self.mode = mode
        assert dc.contains(self.datasets_name), 'Unknown dataset_name: {}'.format(self.datasets_name)
        self.metas = []
        if mode == 'train':
            self.batch_size = cfg.TRAIN.BATCH_SIZE
        else:
            self.batch_size = cfg.TEST.BATCH_SIZE
        self.da_num_classes = dc.get_da_classes(self.cfg.TRAIN.DATASETS)

        source_file = dc.get_source_index(self.datasets_name)[self.mode]
        print(source_file)
        prefix = dc.get_prefix(self.datasets_name)
        with open(source_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                file_name = line.strip()
                img1_path = os.path.join(prefix, 'pre-event', file_name + '_pre_disaster.tif')
                img2_path = os.path.join(prefix, 'post-event', file_name + '_post_disaster.tif')
                label_path = os.path.join(prefix, 'target', file_name + '_building_damage.tif')
                self.metas.append([img1_path, img2_path, label_path])
        self.num = len(lines)

    def __len__(self):
        return self.num

    def __transforms(self, pre_img, post_img, label):

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, label

    def __getitem__(self, idx):
        pre_path, post_path, label_path = self.metas[idx]

        pre_img = io.imread(pre_path)[:, :, 0:3]
        post_img = io.imread(post_path)

        post_img = np.stack((post_img,) * 3, axis=-1)
        clf_label = io.imread(label_path)

        pre_img, post_img, clf_label = self.__transforms(pre_img, post_img, clf_label)
        clf_label = np.asarray(clf_label)

        loc_label = clf_label.copy()
        loc_label[(loc_label != 0) & (loc_label != 255)] = 1

        return pre_img, post_img, loc_label, clf_label


class BDADataset(Dataset):
    def __init__(self, cfg, mode='train', data_list=None):
        self.cfg = cfg
        self.datasets_name = cfg.TRAIN.DATASETS
        self.mode = mode
        assert dc.contains(self.datasets_name), 'Unknown dataset_name: {}'.format(self.datasets_name)
        self.metas = []
        if mode == 'train':
            self.batch_size = cfg.TRAIN.BATCH_SIZE
        else:
            self.batch_size = cfg.TEST.BATCH_SIZE
        self.da_num_classes = dc.get_da_classes(self.cfg.TRAIN.DATASETS)
        self.crop_size = cfg.AUG.INPUT_SIZE
        self.random_hflip = cfg.AUG.RANDOM_HFLIP
        self.random_vflip = cfg.AUG.RANDOM_VFLIP
        self.random_rot = cfg.AUG.RANDOM_ROTATION
        self.mean = dc.get_mean(self.datasets_name)
        self.std = dc.get_std(self.datasets_name)

        prefix = dc.get_prefix(self.datasets_name)
        if data_list is not None:
            lines = data_list
        else:
            source_file = dc.get_source_index(self.datasets_name)[self.mode]
            print(source_file)
            with open(source_file, 'r') as f:
                lines = f.readlines()
        for line in lines:
            file_name = line.strip()
            img1_path = os.path.join(prefix, 'pre-event', file_name + '_pre_disaster.tif')
            img2_path = os.path.join(prefix, 'post-event', file_name + '_post_disaster.tif')
            label_path = os.path.join(prefix, 'target', file_name + '_building_damage.tif')
            self.metas.append([img1_path, img2_path, label_path])

        self.num = len(self.metas)

    def __len__(self):
        return self.num

    def __transforms(self, aug, pre_img, post_img, label):
        if aug:
            pre_img, post_img, label = imutils.random_crop(pre_img, post_img, label, self.crop_size)
            if self.random_hflip:
                pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label)
            if self.random_vflip:
                pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label)
            if self.random_rot:
                pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label)

        pre_img = imutils.normalize_img(pre_img, mean=self.mean, std=self.std)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img, mean=self.mean, std=self.std)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, label

    def __getitem__(self, idx):
        pre_path, post_path, label_path = self.metas[idx]

        pre_img = io.imread(pre_path)[:, :, 0:3]
        post_img = io.imread(post_path)

        post_img = np.stack((post_img,) * 3, axis=-1)
        clf_label = io.imread(label_path)

        if self.mode == 'train':
            pre_img, post_img, clf_label = self.__transforms(True, pre_img, post_img, clf_label)
        else:
            pre_img, post_img, clf_label = self.__transforms(False, pre_img, post_img, clf_label)
        clf_label = np.asarray(clf_label)

        loc_label = clf_label.copy()
        loc_label[(loc_label != 0) & (loc_label != 255)] = 1

        return pre_img, post_img, loc_label, clf_label


class xBDDataset(Dataset):
    def __init__(self, cfg, datasets_name, mode='train'):
        self.cfg = cfg
        # self.datasets_name = cfg.TRAIN.DATASETS
        self.datasets_name = datasets_name
        self.mode = mode
        assert dc.contains(self.datasets_name), 'Unknown dataset_name: {}'.format(self.datasets_name)
        self.metas = []
        if mode == 'train':
            self.batch_size = cfg.TRAIN.BATCH_SIZE
        else:
            self.batch_size = cfg.TEST.BATCH_SIZE
        self.loc_num_classes = 2
        self.da_num_classes = dc.get_da_classes(self.datasets_name)
        self.crop_size = cfg.AUG.INPUT_SIZE
        self.random_hflip = cfg.AUG.RANDOM_HFLIP
        self.random_vflip = cfg.AUG.RANDOM_VFLIP
        self.random_rot = cfg.AUG.RANDOM_ROTATION
        self.mean = dc.get_mean(self.datasets_name)
        self.std = dc.get_std(self.datasets_name)

        source_file = dc.get_source_index(self.datasets_name)[self.mode]
        print(source_file)
        prefix = dc.get_prefix(self.datasets_name)
        with open(source_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                folder, filename = line.strip().split(' ')
                pre_img_path = f'{prefix}/{folder}/images/{filename}_pre_disaster.png'
                post_img_path = f'{prefix}/{folder}/images/{filename}_post_disaster.png'
                pre_mask_path = f'{prefix}/{folder}/targets/{filename}_pre_disaster_target.png'
                post_mask_path = f'{prefix}/{folder}/targets/{filename}_post_disaster_target.png'
                self.metas.append([pre_img_path, post_img_path, pre_mask_path, post_mask_path])
        self.num = len(self.metas)

    def __len__(self):
        return self.num

    def __transforms(self, aug, pre_img, post_img, loc_label, clf_label):
        if aug:
            pre_img, post_img, loc_label, clf_label = imutils.random_crop_bda(pre_img, post_img, loc_label, clf_label,
                                                                              self.crop_size)
            if self.random_hflip:
                pre_img, post_img, loc_label, clf_label = imutils.random_fliplr_bda(pre_img, post_img, loc_label,
                                                                                    clf_label)
            if self.random_vflip:
                pre_img, post_img, loc_label, clf_label = imutils.random_flipud_bda(pre_img, post_img, loc_label,
                                                                                    clf_label)
            if self.random_rot:
                pre_img, post_img, loc_label, clf_label = imutils.random_rot_bda(pre_img, post_img, loc_label,
                                                                                 clf_label)

        pre_img = imutils.normalize_img(pre_img, mean=self.mean, std=self.std)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img, mean=self.mean, std=self.std)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, loc_label, clf_label

    def __getitem__(self, idx):
        pre_path, post_path, loc_label_path, clf_label_path = self.metas[idx]

        pre_img = io.imread(pre_path)
        post_img = io.imread(post_path)
        loc_label = io.imread(loc_label_path)
        clf_label = io.imread(clf_label_path)

        if self.mode == 'train':
            pre_img, post_img, loc_label, clf_label = self.__transforms(True, pre_img, post_img, loc_label, clf_label)
            # clf_label[clf_label == 0] = 255
        else:
            pre_img, post_img, loc_label, clf_label = self.__transforms(False, pre_img, post_img, loc_label, clf_label)
        loc_label = np.asarray(loc_label)
        clf_label = np.asarray(clf_label)

        return pre_img, post_img, loc_label, clf_label


class xBDDataset512(Dataset):
    def __init__(self, cfg, datasets_name, mode='train'):
        self.cfg = cfg
        # self.datasets_name = cfg.TRAIN.DATASETS
        self.datasets_name = datasets_name
        self.mode = mode
        assert dc.contains(self.datasets_name), 'Unknown dataset_name: {}'.format(self.datasets_name)
        self.metas = []
        if mode == 'train':
            self.batch_size = cfg.TRAIN.BATCH_SIZE
        else:
            self.batch_size = cfg.TEST.BATCH_SIZE
        self.loc_num_classes = 2
        self.da_num_classes = dc.get_da_classes(self.datasets_name)
        self.crop_size = cfg.AUG.INPUT_SIZE
        self.random_hflip = cfg.AUG.RANDOM_HFLIP
        self.random_vflip = cfg.AUG.RANDOM_VFLIP
        self.random_rot = cfg.AUG.RANDOM_ROTATION
        self.mean = dc.get_mean(self.datasets_name)
        self.std = dc.get_std(self.datasets_name)

        source_file = dc.get_source_index(self.datasets_name)[self.mode]
        print(source_file)
        prefix = dc.get_prefix(self.datasets_name)
        with open(source_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                pre_img_path, post_img_path, pre_mask_path, post_mask_path = line.strip().split(' ')
                pre_img_path = os.path.join(prefix, pre_img_path)
                post_img_path = os.path.join(prefix, post_img_path)
                pre_mask_path = os.path.join(prefix, pre_mask_path)
                post_mask_path = os.path.join(prefix, post_mask_path)
                self.metas.append([pre_img_path, post_img_path, pre_mask_path, post_mask_path])
        self.num = len(self.metas)

    def __len__(self):
        return self.num

    def __transforms(self, aug, pre_img, post_img, loc_label, clf_label):
        if aug:
            pre_img, post_img, loc_label, clf_label = imutils.random_crop_bda(pre_img, post_img, loc_label, clf_label,
                                                                              self.crop_size)
            if self.random_hflip:
                pre_img, post_img, loc_label, clf_label = imutils.random_fliplr_bda(pre_img, post_img, loc_label,
                                                                                    clf_label)
            if self.random_vflip:
                pre_img, post_img, loc_label, clf_label = imutils.random_flipud_bda(pre_img, post_img, loc_label,
                                                                                    clf_label)
            if self.random_rot:
                pre_img, post_img, loc_label, clf_label = imutils.random_rot_bda(pre_img, post_img, loc_label,
                                                                                 clf_label)

        pre_img = imutils.normalize_img(pre_img, mean=self.mean, std=self.std)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img, mean=self.mean, std=self.std)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, loc_label, clf_label

    def __getitem__(self, idx):
        pre_path, post_path, loc_label_path, clf_label_path = self.metas[idx]

        pre_img = io.imread(pre_path)
        post_img = io.imread(post_path)
        loc_label = io.imread(loc_label_path)
        clf_label = io.imread(clf_label_path)

        if self.mode == 'train':
            pre_img, post_img, loc_label, clf_label = self.__transforms(True, pre_img, post_img, loc_label, clf_label)
            # clf_label[clf_label == 0] = 255
        else:
            pre_img, post_img, loc_label, clf_label = self.__transforms(False, pre_img, post_img, loc_label, clf_label)
        loc_label = np.asarray(loc_label)
        clf_label = np.asarray(clf_label)

        return pre_img, post_img, loc_label, clf_label

# class BDADataset(Dataset):
#     def __init__(self, cfg, mode='train'):
#         self.cfg = cfg
#         self.datasets_name = cfg.TRAIN.DATASETS
#         self.mode = mode
#         assert dc.contains(self.datasets_name), 'Unknown dataset_name: {}'.format(self.datasets_name)
#         self.metas = []
#         if mode == 'train':
#             self.batch_size = cfg.TRAIN.BATCH_SIZE
#         else:
#             self.batch_size = cfg.TEST.BATCH_SIZE
#         self.da_num_classes = dc.get_da_classes(self.cfg.TRAIN.DATASETS)
#
#         source_file = dc.get_source_index(self.datasets_name)[self.mode]
#         print(source_file)
#         prefix = dc.get_prefix(self.datasets_name)
#         with open(source_file, 'r') as f:
#             lines = f.readlines()
#             for line in lines:
#                 file_name = line.strip()
#                 img1_path = os.path.join(prefix, 'pre-event', file_name + '_pre_disaster.tif')
#                 img2_path = os.path.join(prefix, 'post-event', file_name + '_post_disaster.tif')
#                 label_path = os.path.join(prefix, 'target', file_name + '_building_damage.tif')
#                 self.metas.append([img1_path, img2_path, label_path])
#         self.num = len(lines)
#
#         transforms = []
#         if self.mode == 'train':
#             if self.cfg.AUG.RANDOM_CROP:
#                 transforms.append(A.RandomCrop(self.cfg.AUG.INPUT_SIZE[0], self.cfg.AUG.INPUT_SIZE[1]))
#             if self.cfg.AUG.RANDOM_HFLIP:
#                 transforms.append(A.HorizontalFlip())
#             if self.cfg.AUG.RANDOM_VFLIP:
#                 transforms.append(A.VerticalFlip())
#             if self.cfg.AUG.RANDOM_ROTATION:
#                 transforms.append(A.RandomRotate90(p=1.0))
#
#         transforms.extend([
#             A.Normalize(mean=dc.get_mean(self.datasets_name),
#                         std=dc.get_std(self.datasets_name)),
#             ToTensor()
#         ])
#         self.transforms = A.Compose(transforms)
#
#     def __len__(self):
#         return self.num
#
#     def __getitem__(self, idx):
#         pre_path, post_path, label_path = self.metas[idx]
#
#         pre_img = io.imread(pre_path)[:, :, 0:3]
#         post_img = io.imread(post_path)
#         post_img = np.stack((post_img,) * 3, axis=-1)
#         clf_label = io.imread(label_path)
#
#         transformed = self.transforms(image=np.concatenate([pre_img, post_img], axis=2), mask=clf_label)
#         pre_img, post_img = torch.split(transformed["image"], self.cfg.MODEL.in_channels, dim=0)
#         clf_label = transformed["mask"]
#
#         loc_label = clf_label.clone()
#         loc_label[(loc_label != 0) & (loc_label != 255)] = 1
#
#         pkg = tuple([pre_img, post_img, loc_label, clf_label])
#         return pkg
