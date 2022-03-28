# Last edited on 2020-07-17: category, rectum, cancer, id
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
import PIL
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
from albumentations.augmentations import functional as AF
from albumentations.core.transforms_interface import ImageOnlyTransform
import glob
import os
import matplotlib.pyplot as plt
from albumentations import (
    HorizontalFlip, ShiftScaleRotate, CLAHE, RandomRotate90, Resize,RandomCrop, CenterCrop,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, ElasticTransform, Flip, OneOf, Compose, ReplayCompose, KeypointParams
)
from albumentations.pytorch import ToTensorV2
import multiprocessing.dummy as mp

class ToTensorV3(ToTensorV2):
    """Convert image and mask to `torch.Tensor`."""

    def __init__(self, always_apply=True, p=1.0):
        super(ToTensorV3, self).__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        return (torch.from_numpy(img)).unsqueeze(0)

    def apply_to_mask(self, mask, **params):
        return torch.from_numpy(mask.transpose(2, 0, 1))

class CustomDataset(Dataset):
    '''
    # All outputs are in dict form with keys described below:
    # 1) 'image': shape -> torch.Size([1, 7, 256, 256])
    # 2) 'category': type -> <class 'float'>
    # 3) (unused) 'mask' -> shape : torch.Size([3, self.mask_size])
    # 4) 'rectum': shape -> torch.Size([1, self.mask_size])
    # 5) 'cancer': shape -> torch.Size([1, self.mask_size])
    # 6) 'id': type -> <class 'numpy.int64'>
    ## default value for self.mask_size: (4, 128, 128)
    '''
    def __init__(self, csv_file, dimension='3D', if_with_mask=True, image_size = 256, mask_size = (4, 64, 64),
                label_threshold=25, num_slices_3d=7, extension='bmp', transform=None, if_half_precision=False,
                if_replace_path = False, path_to_be_replaced = 'D:/Rectum_exp', path_new = 'mnt/Rectum_exp',
                 if_repr_num = False):

        assert dimension =='3D' or dimension =='2D'
        # self.csv_file = pd.read_csv(csv_file, converters={'repr_num': eval})
        self.csv_file = pd.read_csv(csv_file)
        self.dimension = dimension
        self.extension = '*.' + extension
        self.image_size = image_size
        self.mask_size = mask_size
        self.label_threshold = label_threshold
        self.path_to_be_replaced = path_to_be_replaced
        self.path_new = path_new
        self.prob_transform = 0.9
        self.num_slices_3d = num_slices_3d
        self.if_repr_num = if_repr_num

        if if_replace_path:
            self.process_path = self.get_new_path
        else:
            self.process_path = self.identity

        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.transform_val()

        if dimension =='3D':
            self.process_id = self.process_id_3d
            if if_with_mask:
                self.path_to_dataDict = self.transform_3d_with_mask

                if mask_size == [7, image_size, image_size]:
                    self.postprocess = self.postprocess_with_mask_3d_no_make_size_change
                else:
                    self.postprocess = self.postprocess_with_mask_3d
            else:
                self.path_to_dataDict = self.transform_3d_without_mask
                self.postprocess = self.identity
            self.resize_kernel = 'trilinear'
        elif dimension =='2D':
            print(f'data provider: 2d')
            self.process_id = self.process_id_2d
            self.resize_kernel = 'bilinear'
            if if_with_mask:
                self.path_to_dataDict = self.transform_2d_with_mask

                if mask_size == [image_size, image_size]:
                    self.postprocess = self.postprocess_with_mask_2d_no_make_size_change
                else:
                    self.postprocess = self.postprocess_with_mask_2d

            else:
                self.path_to_dataDict = self.transform_2d_without_mask
                self.postprocess = self.del_replay
        if if_half_precision:
            self.data_type = torch.float16
            self.precision = self.to_half_precision
        else:
            self.data_type = torch.float32
            self.precision = self.identity

    def __len__(self):
        return len(self.csv_file)

    def del_replay(self, dict):
        del dict['replay']
        return dict

    def transform_val(self):
        return ReplayCompose([CenterCrop(self.image_size, self.image_size, always_apply=True, p=1),
            ToTensorV3(always_apply=True)], p=1)

    def identity(self, x):
        return x

    def get_new_path(self, str):
        str = str.replace(self.path_to_be_replaced, self.path_new)
        return str.replace('\\', '/')

    def process_id_2d(self, idx):
        return self.csv_file.iloc[idx, 2].split(os.sep)[-1].split('.')[0]

    def process_id_3d(self, idx):
        # return self.csv_file.iloc[idx, 2].split(os.sep)[0].split('/')[-1]
        return self.csv_file.iloc[idx, 1]

    def transform_2d_with_mask(self, path_img, path_mask):
        # tuple 형태로 해서 transform_2d_without_mask 랑 하나로 처리할 수 있음. 나중에 바꿀것.
        img = np.array(Image.open(path_img), np.uint8)
        mask = np.array(Image.open(path_mask), np.uint8)
        return self.transform(image = img, mask = mask)

    def transform_2d_without_mask(self, path_img, path_mask):
        return self.transform(image = np.array(Image.open(path_img), np.uint8))

    def transform_3d_with_mask(self, path_imgs, path_masks):
        # tuple 형태로 해서 transform_3d_without_mask 랑 하나로 처리할 수 있음. 나중에 바꿀것.
        path_list_img = sorted(glob.glob(os.path.join(path_imgs, self.extension)))
        path_list_mask = sorted(glob.glob(os.path.join(path_masks, self.extension)))

        augmented = self.transform_2d_with_mask(path_list_img[0], path_list_mask[0])
        stack_dict ={}
        stack_dict['image'] = torch.zeros((1, self.num_slices_3d, self.image_size, self.image_size))

        stack_dict['mask'] = torch.zeros((3, self.num_slices_3d, augmented['mask'].size(-2), augmented['mask'].size(-1)))
        stack_dict['image'][:, 0, :, :] = augmented['image']
        stack_dict['mask'][:, 0, :, :] = augmented['mask']

        for i in range(1, self.num_slices_3d):
            augmented_temp = self.transform.replay( augmented['replay'],
                                        image=np.array(Image.open(path_list_img[i]), np.uint8),
                                        mask=np.array(Image.open(path_list_mask[i]), np.uint8) )
            stack_dict['image'][:, i, :, :] = augmented_temp['image']
            stack_dict['mask'][:, i, :, :] = augmented_temp['mask']

        return stack_dict

    def transform_3d_without_mask(self, path_imgs, path_masks):
        path_list_img = sorted(glob.glob(os.path.join(path_imgs, self.extension)))

        augmented = self.transform_2d_without_mask(path_list_img[0], 0)
        stack_dict = {}
        stack_dict['image'] = torch.zeros((1, self.num_slices_3d, self.image_size, self.image_size))
        stack_dict['image'][:, 0, :, :] = augmented['image']

        for i in range(1, self.num_slices_3d):
            # print(f'image: {np.array(Image.open(path_list_img[i]), np.uint8).shape}')
            # plt.imshow(np.array(Image.open(path_list_img[i]), np.uint8))
            # plt.show()
            augmented_temp = self.transform.replay(augmented['replay'],
                                            image=np.array(Image.open(path_list_img[i]), np.uint8) )

            stack_dict['image'][:, i, :, :] = augmented_temp['image']

        return stack_dict

    def postprocess_with_mask_3d(self, dict):
        dict['mask'] = (dict['mask']).type(torch.float32)
        dict['mask'] = (F.interpolate(dict['mask'].unsqueeze(0), self.mask_size, mode=self.resize_kernel,
                                      align_corners=True)).squeeze(0)
        dict['mask'][dict['mask'] < self.label_threshold] = 0.0
        dict['mask'][dict['mask'] >= self.label_threshold] = 1.0
        dict['rectum'] = dict['mask'][1:2, ::]
        dict['cancer'] = dict['mask'][2:3, ::]
        return dict

    def postprocess_with_mask_3d_no_make_size_change(self, dict):
        dict['mask'] = (dict['mask']).type(torch.float32)
        dict['mask'][dict['mask'] < self.label_threshold] = 0.0
        dict['mask'][dict['mask'] >= self.label_threshold] = 1.0
        dict['rectum'] = dict['mask'][1:2, ::]
        dict['cancer'] = dict['mask'][2:3, ::]
        return dict

    def postprocess_with_mask_2d_no_make_size_change(self, dict):
        dict['mask'] = (dict['mask']).type(torch.float32)
        dict['mask'][dict['mask'] < self.label_threshold] = 0.0
        dict['mask'][dict['mask'] >= self.label_threshold] = 1.0
        dict['rectum'] = dict['mask'][1:2, ::]
        dict['cancer'] = dict['mask'][2:3, ::]
        del dict['replay']
        return dict

    def postprocess_with_mask_2d(self, dict):
        dict['mask'] = (dict['mask']).type(torch.float32)
        dict['mask'] = (F.interpolate(dict['mask'].unsqueeze(0), self.mask_size, mode=self.resize_kernel,
                                          align_corners=True)).squeeze(0)
        dict['mask'][dict['mask'] < self.label_threshold] = 0.0
        dict['mask'][dict['mask'] >= self.label_threshold] = 1.0
        dict['rectum'] = dict['mask'][1:2, ::]
        dict['cancer'] = dict['mask'][2:3, ::]
        del dict['replay']
        return dict

    def to_half_precision(self, dict):
        for key in dict.keys():
            if key == 'image' or key == 'rectum' or key == 'cancer':
                dict[key] = dict[key].half()
        return dict

    def __getitem__(self, idx):

        path_image = self.process_path(self.csv_file.iloc[idx, 2])
        path_mask = self.process_path(self.csv_file.iloc[idx, 3])
        # try:
        dataDict = self.path_to_dataDict(path_image, path_mask)
        # except:
        #     if float(self.csv_file.iloc[idx, 0]) == 0:
        #         ts = 'T2'
        #     elif float(self.csv_file.iloc[idx, 0]) == 1:
        #         ts = 'T3'
        #     print(f'T-stage: {ts}, patient id: {self.process_id(idx)}')
        dataDict['id'] = self.process_id(idx)
        dataDict['category'] = float(self.csv_file.iloc[idx, 0])
        dataDict['image'] = (dataDict['image']).type(torch.float32)
        if self.dimension == '3D':
            repr_num = self.csv_file.at[idx, 'repr_num']
            repr_num = [int(s) for s in repr_num.replace('[','').replace(']','').split(', ')]
            
            #rint(f'repr_num: {repr_num}, type: {type(repr_num)}')
            
            dataDict['repr_num'] = torch.zeros(7)
            dataDict['repr_num'][repr_num]=1
            #rint(f'after: {dataDict["repr_num"]}')a
        # dataDict['repr_num'] = list(temp_list)

        return self.precision(self.postprocess(dataDict))
