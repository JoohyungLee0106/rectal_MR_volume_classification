from albumentations import (
    HorizontalFlip, ShiftScaleRotate, CLAHE, RandomRotate90, Resize,RandomCrop, CenterCrop,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, ElasticTransform, Flip, OneOf, Compose, ReplayCompose, KeypointParams
)
from albumentations.pytorch import ToTensorV2
import random
import cv2
import torch
from albumentations.augmentations import functional as AF
from albumentations.core.transforms_interface import ImageOnlyTransform

class ToTensorV3(ToTensorV2):
    """Convert image and mask to `torch.Tensor`."""

    def __init__(self, always_apply=True, p=1.0):
        super(ToTensorV3, self).__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        return (torch.from_numpy(img)).unsqueeze(0)

    def apply_to_mask(self, mask, **params):
        return torch.from_numpy(mask.transpose(2, 0, 1))

###########################################################################################################
def augmentation(IMAGE_SIZE, if_test = False):
    if if_test:
        # return ReplayCompose([
        #     ElasticTransform(alpha=10.0 * random.random(), border_mode=cv2.BORDER_CONSTANT, value=0, alpha_affine=0,
        #                      p=0.5),
        #     OneOf([
        #         OpticalDistortion(distort_limit=0.02, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
        #         GridDistortion(distort_limit=0.01, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5)
        #     ], p=0.5),
        #     ShiftScaleRotate(shift_limit=(0.02, 0), scale_limit=(-0.02, 0.05), rotate_limit=0, p=0.5,
        #                      border_mode=cv2.BORDER_CONSTANT),
        #
        #     CenterCrop(IMAGE_SIZE, IMAGE_SIZE, always_apply=True, p=1),
        #
        #     HorizontalFlip(),
        #     CLAHE(clip_limit=(1, 2), p=0.5),
        #     # IAAAdditiveGaussianNoise(p=0.6),
        #     RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.5),
        #     # OneOf([
        #     #     MotionBlur(blur_limit=4, p=0.9),
        #     #     MedianBlur(blur_limit=3, p=0.5),
        #     #     Blur(blur_limit=3, p=0.5),
        #     # ], p=0.1),
        #
        #     # ResizeV2(self.image_size, self.image_size, interpolation=cv2.INTER_LANCZOS4, always_apply=True, p=1),
        #     ToTensorV3(always_apply=True)
        # ], p=0.9)
        return ReplayCompose([
            CenterCrop(IMAGE_SIZE, IMAGE_SIZE, always_apply=True, p=1),
            ToTensorV3(always_apply=True)
        ], p=1)

    else:
        return ReplayCompose([
                ElasticTransform(alpha=20.0 * random.random(), border_mode=cv2.BORDER_CONSTANT, value=0, alpha_affine=5,
                                 p=0.9),
                OneOf([
                    OpticalDistortion(distort_limit=0.03, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.9),
                    GridDistortion(distort_limit=0.02, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5)
                ], p=0.5),
                ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=0.7,
                                 border_mode=cv2.BORDER_CONSTANT),

                RandomCrop(IMAGE_SIZE, IMAGE_SIZE, always_apply=True, p=1),

                HorizontalFlip(),
                CLAHE(clip_limit=(0, 2), p=0.8),
                # IAAAdditiveGaussianNoise(p=0.6),
                RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.8),
                OneOf([
                    MotionBlur(blur_limit=4, p=0.9),
                    MedianBlur(blur_limit=3, p=0.5),
                    Blur(blur_limit=3, p=0.5),
                ], p=0.1),

                # ResizeV2(self.image_size, self.image_size, interpolation=cv2.INTER_LANCZOS4, always_apply=True, p=1),
                ToTensorV3(always_apply=True)
            ], p=0.9)