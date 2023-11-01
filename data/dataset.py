__author__ = 'Titi Wei'
import torch
import torchvision
from torch.utils.data.dataset import Dataset
from torchvision.transforms import functional as TF
from torch.nn import functional as F
import numpy as np
import os, glob
import random

import sys
sys.path.append('data')
import utils

class BaseDataset(Dataset):
    def __init__(self):
        transforms = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ]
        self.transform = torchvision.transforms.Compose(transforms)

        self.totensor = torchvision.transforms.ToTensor()

    def normalize(self, x):
        return (x - x.min()) / (x.max() - x.min())
    
    def to_one_hot(self, x, num_classes):
        x = torch.from_numpy(x).type(torch.long)
        x = F.one_hot(x, num_classes=num_classes).type(torch.float32)
        x = x.permute(2, 0, 1)
        return x

class CTDataset(BaseDataset):
    def __init__(self, folder, list_path, num_classes=72, train=True):
        super().__init__()
        self.num_classes = num_classes
        self.train = train

        # open list and glob all ct slice paths
        self.List = utils.open_txt(list_path)
        self.paths = []
        for number in self.List:
            path = os.path.join(folder, number)
            ct_paths = glob.glob(os.path.join(path, 'CT', '*.npy'))
            self.paths += ct_paths

    def __getitem__(self, index):
        ct_path = self.paths[index]
        seg_path = ct_path.replace('CT', 'Seg')
        ct = np.load(ct_path)
        seg = np.load(seg_path)

        # Normalization and transform
        ct = self.normalize(ct)
        ct = self.transform(ct).type(torch.float32)
        # Segmentation class to one hot
        seg = self.to_one_hot(seg, self.num_classes)

        # data augmentation rotate
        if self.train:
            angle = random.choice([0, 90, 180, 270])
            ct = TF.rotate(ct, angle)
            seg = TF.rotate(seg, angle)

        return ct, seg

    def __len__(self):
        return len(self.paths)
    
class PredictDataset(BaseDataset):
    def __init__(self, folder, num_classes=72):
        super().__init__()
        self.num_classes = num_classes

        self.paths = glob.glob(os.path.join(folder, 'CT', '*.npy'))
        self.paths.sort()

    def __getitem__(self, index):
        ct_path = self.paths[index]
        ct = np.load(ct_path)

        # Normalization and transform
        ct = self.normalize(ct)
        ct = self.transform(ct).type(torch.float32)

        return ct

    def __len__(self):
        return len(self.paths)
    
if __name__ == '__main__':
    # dataset = CTDataset('/root/VGHTC/No_IV_197_preprocessed', 'trainList.txt', num_classes=72, train=True)
    dataset = PredictDataset('/root/VGHTC/No_IV_197_preprocessed/000074623G')
    print(dataset.__len__())
    for ct in dataset:
        print(ct.shape)
        break