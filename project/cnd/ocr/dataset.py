import os
import re
import numpy as np
import cv2

#from torch.utils.data import Dataset


class OcrDataset():#Dataset):
    def __init__(self, data_path, target_path, transforms=None):
        # TODO: Here you can create samples from dirs and initialize transfroms
        def load(x):
            fnames = sorted([os.path.join(data_path,fname) for fname in os.listdir(data_path) if
                             re.fullmatch('[a-zA-Z]\d{3}[a-zA-Z]{2} \d+\.bmp', fname) is not None])
            return fnames
        self.data = load(data_path)
        self.target = load(target_path)
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = cv2.imread(self.data[idx])
        t = os.path.split(self.target[idx])[-1][:5] #np.load(self.target[idx])#
        # TODO: Apply transforms to img and target if it necessary
        for trns in self.transforms:
            img = trns(img)

        return img, t


