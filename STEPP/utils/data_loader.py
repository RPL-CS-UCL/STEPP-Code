import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset
import numpy as np

class FeatureDataset:
    def __init__(self, feature_dir, transform=None, target_transform=None, batch_size=None) -> None:
        self.feature_dir = feature_dir
        self.transform = transform
        self.batch_size = batch_size
        self.target_transform = target_transform        
        self.avg_features = np.load(self.feature_dir)

    def __len__(self) -> int:
        return len(self.avg_features)

    def __getitem__(self, idx: int):
        
        if self.batch_size:
            feature = self.avg_features[idx:idx+self.batch_size]
            print(feature.shape)
        else:
            feature = self.avg_features[idx]
        if self.transform:
            feature = self.transform(feature)
        if self.target_transform:
            feature = self.target_transform(feature)
        return feature