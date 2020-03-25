## Author: Egor Dudyrev

import torch
import numpy as np
import skimage
import cv2
from toolz import compose


class ImageNormalization(object):
    def __call__(self, image):
        return image / 255.


class ImageScaling(object):
    def __init__(self, dsize):
        self._dsize = dsize

    def __call__(self, image, dsize=None):
        dsize = self._dsize if dsize is None else dsize
        return cv2.resize(image, (dsize[1], dsize[0]))


class ToType(object):
    def __init__(self, type_=np.float32):
        self._type_ = type_

    def __call__(self, image, type_=None):
        type_ = self._type_ if type_ is None else type_
        return image.astype(type_)


class ToGreyScale(object):
    def __call__(self, image, strategy='sklearn'):
        if len(image.shape)==2:
            return np.reshape(image, (image.shape[0], image.shape[1], 1))
        gscale = image
        if strategy == 'sklearn':
            # Y = 0.2125 R + 0.7154 G + 0.0721 B.
            # These weights represent human colour perception better than just mean values. According to sklean
            gscale[:, :, 0] = gscale.dot([0.2125, 0.7154, 0.0721])
        elif strategy == 'mean':
            gscale[:, :, 0] = gscale.dot([1, 1, 1])/3
        return gscale[:, :, :1]


class RandomCrop(object):
    def __init__(self, size):
        self._size = size

    def __call__(self, img, size=None, seed=None):
        size = self._size if size is None else size
        assert size[0] < img.shape[0], 'Size height is bigger than image'
        assert size[1] < img.shape[1], 'Size width is bigger than image'

        y_max = img.shape[0] - size[0]
        x_max = img.shape[1] - size[1]

        np.random.seed(seed)
        x, y = [np.round(np.random.uniform(low=0, high=lim), 0).astype(int) for lim in [x_max, y_max]]
        return img[y:y + size[0], x:x + size[1]]


class ImageRotation(object):
    def __init__(self, angle='random'):
        self._angle = angle

    def __call__(self, img, angle=None, seed=None):
        np.random.seed(seed)
        angle = self._angle if angle is None else angle
        if angle == 'random':
            angle = np.random.uniform(-5, 5)#+np.random.choice([0,180])
        return skimage.transform.rotate(img, angle)


class ChangeBrightnessContrast(object):
    def __init__(self, brightness='random', contrast='random'):
        self._brightness = brightness
        self._contrast = contrast

    def __call__(self, img, brightness=None, contrast=None, seed=None):
        if img.max() <= 1:
            img = img*255

        brightness = self._brightness if brightness is None else brightness
        contrast = self._contrast if contrast is None else contrast

        np.random.seed(seed)
        if brightness == 'random':
            brightness = np.round(np.random.uniform(-100, 100), 0).astype(int)
        if contrast == 'random':
            contrast = np.random.uniform(0.3, 1.5)

        new_img = ImageNormalization()(cv2.convertScaleAbs(img, alpha=contrast, beta=brightness))
        return new_img


class ToTensor(object):
    def __call__(self, image):
        image = image.astype(np.float32)
        image = np.swapaxes(image, 0, 1)
        image = np.swapaxes(image, 0, 2)
        return torch.from_numpy(image)


def get_transforms(image_size):
    transform = compose(ToTensor(),
                        RandomCrop(image_size), ToGreyScale(), ToType(), #ChangeBrightnessContrast(),
                        ImageScaling((image_size[0]+10, image_size[1]+10)), ImageNormalization())# USE COMPOSE TO APPLY ALL YOUR TRANSFORMS
    return transform
