import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split

class fake_Autoencoder():
    def __init__(self):
        # define size
        self.window_size = 3
        self.box_size = 10
        name_list = os.listdir('data/train/tensor/')
        tensor = torch.load('data/train/tensor/' + name_list[0])
        self.size_z = len(tensor[0])
        self.size_c = len(tensor[0][0])
        self.bottleneck = int(self.size_z/self.box_size)*2

    def compress(self, data):
        mean_x = data[0][:,:,1] #0 is terrain
        mean_z = data[0][:,:,2]
        sig_xz = data[0][:,:,3]
        idx = np.arange(0,self.size_z, int(self.box_size))
        pred = np.zeros((len(idx)*2)) # only consider 2 channels at the moment
        for i in range(len(idx)):
            pred[i] = np.mean(mean_x[:,idx[i]:idx[i] + self.box_size])
            pred[len(idx)+i] = np.mean(mean_z[:,idx[i]:idx[i] + self.box_size])
        return pred

    def window(self, data, center=-1):
        size_x = len(data)
        size_z = len(data[0])
        size_c = len(data[0][0])
        if center == -1:
            center = np.random.randint(0,size_x)
        start = int(max(center - self.window_size, 0))
        end = int(min(center + self.window_size, size_x))

        window = np.ones((self.window_size*2,size_z,size_c))*0
        if start == 0: #if touching the left border
            window[0:end,:] = data[start:end,:]
        elif end == size_x: #if touching the right border
            window[self.window_size*2 - (end-start):self.window_size*2,:] = data[start:end,:]
        else: #if not touching anything
            window = data[start:end,:]
        return window

def load_tensor(path):
    name_list = os.listdir(path)
    name_list.sort()
    tensor_list = []
    for name in name_list:
        tensor = torch.load(path + name)
        tensor_list.append(tensor)
    return np.array(tensor_list)
