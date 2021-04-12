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
        self.size_c = len(tensor)
        self.size_x = len(tensor[0])
        self.size_z = len(tensor[0][0])
        self.bottleneck = int(self.size_z/self.box_size)*2 + 2 #for terrain

    def model_test(self, data, position):
        window = self.window(data, position[0])
        terrain = self.compress_terrain(window, position[1])
        wind = self.compress_wind(window)
        return np.concatenate((terrain, wind))

    def window(self, data, center):
        start_x = int(max(center - self.window_size, 0))
        end_x = int(min(center + self.window_size, self.size_x))
        window = np.zeros((self.size_c,self.window_size*2,self.size_z))
        fill_in = data[:,start_x:end_x,:]
        # touching the left border
        if start_x == 0:
            window[:,self.window_size*2-end_x::,:] = fill_in
            for i in range(2*self.window_size-len(fill_in[0])):
                window[:,i,:] = fill_in[:,0,:]

        # touching the right border
        elif end_x == self.size_x:
            window[:,0:end_x-start_x,:] = fill_in
            for i in range(2*self.window_size-len(fill_in[0])):
                window[:,2*self.window_size-i-1,:] = fill_in[:,-1,:]

        # if not touching anything¨
        else:
            #print('no touch')
            window = fill_in

        window = torch.tensor(window)
        return window

    def compress_wind(self, data):
        mean_x = data[-3,:,:]
        mean_z = data[-2,:,:]
        sig_xz = data[-1,:,:]

        idx = np.arange(0,self.size_z, int(self.box_size))
        pred = np.zeros((len(idx)*2)) # only consider 2 channels at the moment
        # wind
        for i in range(len(idx)):
            pred[i] = torch.mean(mean_x[:,idx[i]:idx[i] + self.box_size])
            pred[len(idx)+i] = torch.mean(mean_z[:,idx[i]:idx[i] + self.box_size])
        return pred

    def compress_terrain(self, data, altitude):
        distance = data[1,:,:]  #0 is terrain
        bearing = data[2,:,:]
        i = int(self.window_size/2)
        j = min(int(altitude), self.size_z-1)
        return [distance[i,j].item(), bearing[i,j].item()]

def load_tensor(path):
    name_list = os.listdir(path)
    name_list.sort()
    tensor_list = []
    for name in name_list:
        tensor = torch.load(path + name)
        tensor_list.append(tensor)
    return np.array(tensor_list)
