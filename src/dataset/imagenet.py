import torchvision
import numpy as np

from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image


class ImageNet(Dataset):
    def __init__(self, path, split, transform=None):
        self.path = path
        self.split = split
        self.transform = transform
        self.dataset = torchvision.datasets.ImageNet(path, split=split, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = np.array(img)
        return img, label


class TinyImageNet(Dataset):
    def __init__(self, split, transform=None):
        self.split = split
        self.transform = transform
        self.dataset = load_dataset('zh-plus/tiny-imagenet')[split]

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
