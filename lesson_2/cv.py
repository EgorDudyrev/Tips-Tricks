import numpy as np

from dpipe.medim.preprocessing import get_greatest_component
from dpipe.medim.box import mask2bounding_box, add_margin, box2slices
from dpipe.medim.utils import apply_along_axes

from skimage.filters import gaussian
from skimage.morphology import binary_opening, disk

from scipy.ndimage import convolve
from scipy.interpolate import UnivariateSpline

from functools import partial


def get_spine_bbox(x):
    s = (x > 250).sum(axis=(2))
    # размытие
    s_ = gaussian(s, sigma=3, preserve_range=True)
    s_ = s_ / s_.max() > 0.45
    # открытие морфология
    s_ = binary_opening(s_, disk(4))
    # наиб связ компонента
    s_ = get_greatest_component(s_)

    # bbox = mask2bounding_box(s_)
    # s_[1][0]-=10
    bbox = mask2bounding_box(s_)
    bbox = add_margin(bbox, [20, 0])
    b = bbox.copy()
    b[0][0] -= 20

    return b


def cut_spine(x):
    b = get_spine_bbox(x)
    return x[box2slices(b)]


def get_mask_full_spine(y):
    y_ = y > 120
    b_r = 3
    b_h = 50
    #цилиндр высоты b_h и радиуса b_r
    bone_cylinder = np.stack([disk(b_r) for i in range(b_h)])
    y__ = convolve(y_.astype(float), bone_cylinder) / bone_cylinder.sum()
    y_b = get_greatest_component(y__ > 0.2)

    return y_b


def fill_line(line):
    l = np.zeros(line.shape)
    a, b = np.argmax(line, axis=0), line.shape[0] - np.argmax(line[::-1, ...], axis=0)
    for i, (a_, b_) in enumerate(zip(a, b)):
        if not line[..., i].any():
            continue
        l[a_:b_, i] = True

    return l


def find_spinal_cord(y):
    y_spine = get_mask_full_spine(y)
    # зальем "по y"
    y_ = apply_along_axes(fill_line, y_spine, axes=(1, 2)).astype(bool)
    # возьмем неплотное внутри
    y_n = (y < 100) & y_
    # свернем с цилиндром 5 default
    mask = np.stack([disk(5) for i in range(20)])
    y__ = convolve(y_n.astype(int), mask) / mask.sum()
    y_f = get_greatest_component(y__ > 0.85)

    return y_f


def find_centre_for_slice(s):
    sum_ = s.sum()
    if not sum_:
        return [np.NaN, np.NaN]
    
    a = s.sum(axis=1) / sum_
    b = s.sum(axis=0) / sum_

    a = (np.array([i for i in range(len(a))]) * a).sum()
    b = (np.array([i for i in range(len(b))]) * b).sum()

    return [np.round(a).astype(int), np.round(b).astype(int)]


def find_centres(y_sc):
    return np.stack([find_centre_for_slice(s) for s in y_sc])