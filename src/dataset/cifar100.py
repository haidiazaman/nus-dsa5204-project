import torchvision
import numpy as np

from torch.utils.data import Dataset


class CIFAR100(Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = path
        self.train = train
        self.transform = transform
        self.dataset = torchvision.datasets.CIFAR100(path, train=train, transform=transform, download=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = np.array(img)
        return img, label
