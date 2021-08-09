import os
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm
from torchsummary import summary

# data augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

# pretrained models
import torchvision
from torchvision import models, transforms


def resize_img(path, new_path, size):
    desired_size = size
    img = cv2.imread(path)
    old_size = img.shape[:2]
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    
    img = cv2.resize(img, (new_size[1], new_size[0]))
    
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)
    
    cv2.imwrite(new_path, new_img)
    

def center_crop(path, new_path, size):
    dim = (size, size)
    img = cv2.imread(path)
    width, height = img.shape[1], img.shape[0]
    
    crop_width = dim[0] if dim[0] < width else height
    crop_height = dim[1] if dim[1] < height else width
    
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2) 
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    
    cv2.imwrite(new_path, crop_img)
    
    
def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img / 255.).float()
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = normalize(img)
    
    
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, hidden_x1, hidden_x2, y):
        # euclidian distance
        diff = hidden_x1 - hidden_x2
        dist_sq = torch.sum(torch.pow(diff, 2), 2)
        dist = torch.sqrt(dist_sq + self.eps)

        mdist = self.margin - dist
        dist = torch.maximum(mdist, torch.zeros(mdist.shape))
        loss = ((y * dist_sq) + ((1 - y) * torch.pow(dist, 2))) / 2
        # average loss over a batch
        avg_loss = torch.sum(loss)/hidden_x1.shape[1]
        return avg_loss