import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.io import read_image, ImageReadMode
import torch as T
from sarpy.io.complex.sicd import SICDReader

import os

__all__ = ["SICDDataset", "ImageDataset"]

class SICDDataset(Dataset):
    def __init__(self, file_dir, chip=True, file_hw=500, transform=None, sample_rate=1):
        self.file_dir = file_dir
        self.files = [file for file in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, file)) and file.endswith(("ntf", "nitf"))]
        self.chip = chip
        self.hw = file_hw
        self.transform = transform
        self.num_files = len(self.files)
        self.sample_rate = sample_rate
        
    def __len__(self):
        return self.num_files*self.sample_rate

    def __getitem__(self, idx):
        self.file_path = os.path.join(self.file_dir, self.files[idx%self.num_files])
        reader = SICDReader(self.file_path)
        data_size = reader.data_size
        if self.chip:
            start_x = np.random.choice(data_size[0]-self.hw)
            start_y = np.random.choice(data_size[1]-self.hw)
            sicd = reader[start_x:start_x+self.hw,start_y:start_y+self.hw]
        else:
            sicd = reader[:]
        sicd = T.tensor(sicd, dtype=T.complex64)
        if self.transform:
            sicd = self.transform(sicd)
        return sicd[None, :, :]

class ImageDataset(Dataset):
    def __init__(self, file_dir, transform=None, sample_rate=1):
        self.file_dir = file_dir
        self.transform = transform
        self.files = [file for file in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, file))]
        self.num_files = len(self.files)
        self.sample_rate = sample_rate

    def __len__(self):
        return self.num_files*self.sample_rate

    def __getitem__(self, idx):
        img_path = os.path.join(self.file_dir, self.files[idx%self.num_files])
        image = read_image(img_path, mode=ImageReadMode.GRAY)
        if self.transform:
            image = self.transform(image.type(dtype=T.float))
        return T.complex(image, T.zeros(image.shape))