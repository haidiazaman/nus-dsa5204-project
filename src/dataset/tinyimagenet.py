import numpy as np

from datasets import load_from_disk
from torch.utils.data import Dataset
from PIL import Image


class TinyImageNet(Dataset):
    def __init__(self, split, path='./data/tinyimagenet', transform=None):
        self.split = split
        self.transform = transform
        self.dataset = load_from_disk(path)[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]['image'], self.dataset[idx]['label']
        if np.shape(img) != (64, 64, 3):
            img = np.dstack([img, img, img])
            img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label