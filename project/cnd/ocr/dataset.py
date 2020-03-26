## Author: Egor Dudyrev

import os
import re
import imageio


from torch.utils.data import Dataset


class OcrDataset(Dataset):
    def __init__(self, data_path, target_path=None, transforms=None):
        # TODO: Here you can create samples from dirs and initialize transfroms
        def load(x):
            fnames = sorted([os.path.join(data_path, fname) for fname in os.listdir(data_path)])
            return fnames
        self.data = load(data_path)
        self.target = load(target_path if target_path is not None else data_path)
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = imageio.imread(self.data[idx])
        text = os.path.split(self.target[idx])[-1].replace('.bmp', '').replace(' ', '')
        #print('TEXT', text, self.target[idx])

        # TODO: Apply transforms to img and target if it necessary
        img = self.transforms(img) if self.transforms is not None else img

        return {"image": img[None],
                "text": text}
