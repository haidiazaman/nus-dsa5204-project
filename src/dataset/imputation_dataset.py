from torch.utils.data import Dataset
import torch
import pandas as pd

def get_housing_dataset(path):
    return pd.read_csv(path).dropna()

class HousingDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.dropna()  # Remove NaN
        self.mask = ~dataframe.isna()   # Mask indicating present values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.tensor(self.data.iloc[idx].values, dtype=torch.float32)
        mask = torch.tensor(self.mask.iloc[idx].values, dtype=torch.float32)
        return data, mask