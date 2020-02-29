import torch
import numpy as np


class ImageNormalization(object):
    def __call__(self, image):
        return image / 255.


class ToTensor(object):
    def __call__(self, image):
        image = image.astype(np.float32)
        return torch.from_numpy(image)

#TODO: Your transforms here
# Basic transforms:
# - Scale transform, to change size of input image
# - ToType transform, change type of image (usually image has type uint8)
# - ToTensor transform, move image to PyTorch tensor (to GPU?)
# - ImageNormalization, change scale of image from [0., ..., 255.] to [0., ..., 1.] (all float)
# Also you can add augmentations:
# - RandomCrop
# - RandomFlip
# - Brightness and Contrast augmentation
# Or any other, you can use https://github.com/albumentations-team/albumentations or any other lib