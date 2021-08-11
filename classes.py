import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from functions import *


### siamese network model ###
class SiameseNetwork(nn.Module):
    """
    Train network with two images at once to 
    perform binary classification.
    """
    def __init__(self, features):
        super(SiameseNetwork, self).__init__()
        # individual
        self.features = features
        # shared
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(8192, 1024)
        self.linear2 = nn.Linear(1024, 1)
        
    def forward(self, x1, x2):
        x1_features = self.features(x1)
        x2_features = self.features(x2)
        x = torch.cat([x1_features, x2_features], 1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x.squeeze()


### datasets ###
class ArtistDataset(Dataset):
    """
    Pass one image at a time and output multiclass label.
    """
    def __init__(self, df, label_dict):
        self.df = df
        
        self.label_dict = label_dict
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img = load_image(row['full_path'])
        
        label = torch.tensor(self.label_dict[row['artist']])
        
        return img, label
    
    
class ArtistPairsDataset(Dataset):
    """
    Pass two images at a time and output binary class.
    """
    def __init__(self, df):
        self.df = df
        
        self.remap = {False: 0, True: 1}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_1 = load_image(row['filename_1'])
        img_2 = load_image(row['filename_2'])
        
        label = torch.tensor(self.remap[row['same_artist']]).float()
        
        return img_1, img_2, label