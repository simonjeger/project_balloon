import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

from PIL import Image
import cv2

class custom_data(Dataset):
    def __init__(self, path):
        name_list = os.listdir(path)
        name_list.sort()
        self.data = []
        for name in name_list:
            img = cv2.imread(path + name)
            img = cv2.resize(img, (130,150), interpolation = cv2.INTER_AREA)
            img = Image.fromarray(img).convert('RGB')  #img as opencv

            arr = np.array(img).transpose(-1, 0, 1) #because pytorch reads in images as Channel x With x Hight
            arr = arr/255
            self.data.append(torch.tensor(arr).float()) # because there are no labels and only one subfolder of data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class wind_data(Dataset):
    def __init__(self, path):
        name_list = os.listdir(path)
        name_list.sort()
        self.data = []
        for name in name_list:
            tensor = torch.load(path + name)
            tensor = np.array(tensor).transpose(-1, 0, 1)
            self.data.append(torch.tensor(tensor).float())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
