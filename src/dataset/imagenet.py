import torch
import torchvision
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
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
        img = img.resize((256, 256), resample=0)
        img = np.array(img)
        return img, label
