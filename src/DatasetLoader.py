import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

class DatasetLoader(Dataset):
    ## Ref: https://discuss.pytorch.org/t/input-numpy-ndarray-instead-of-images-in-a-cnn/18797/2
    ## data needs shape of [batch__size, c, h, w]
    def __init__(self, data, annotations_file, transform=None, target_transform=None):
        self.data = torch.from_numpy(data).float()
        self.annotations_file = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.annotations_file[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y