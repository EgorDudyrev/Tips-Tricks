import cv2
import numpy as np

from torch.utils.data import Dataset


class OcrDataset(Dataset):
    def __init__(self, data_path, target_path, transforms=None):
        # TODO: Here you can create samples from dirs and initialize transfroms
        self.data = load(data_path)
        self.target = load(target_path)
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = cv2.imread(self.data[idx])
        t = np.load(self.target[idx])
        # TODO: Apply transforms to img and target if it necessary

        return img, t


