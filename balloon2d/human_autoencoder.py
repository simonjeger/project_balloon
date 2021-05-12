import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

class HAE():
    def __init__(self):
        # define size
        name_list = os.listdir(yaml_p['data_path'] + 'train/tensor/')
        tensor = torch.load(yaml_p['data_path'] + 'train/tensor/' + name_list[0])
        self.size_c = len(tensor)
        self.size_x = len(tensor[0])
        self.size_z = len(tensor[0][0])

        self.window_size = yaml_p['window_size']
        self.window_size_total = 2*self.window_size + 1
        self.box_size = int(self.size_z/yaml_p['bottleneck'])

        self.bottleneck_terrain = 2
        self.bottleneck_wind = int(self.size_z/self.box_size)*2
        self.bottleneck = self.bottleneck_terrain + self.bottleneck_wind

    def window(self, data, center):
        window = np.zeros((len(data),self.window_size_total,self.size_z))
        data_padded = np.zeros((len(data),self.size_x+2*self.window_size,self.size_z))
        data_padded[:,self.window_size:-self.window_size,:] = data

        for i in range(self.window_size):
            data_padded[:,i,:] = data_padded[:,self.window_size,:]
            data_padded[:,-(i+1),:] = data_padded[:,-(self.window_size+1),:]

        start_x = int(center)
        end_x = int(center + self.window_size_total)

        window = data_padded[:,start_x:end_x,:]
        window = torch.tensor(window)
        return window

    def compress(self, data, position):
        window = self.window(data, position[0])
        terrain = self.compress_terrain(window, position)
        wind = self.compress_wind(window)
        return np.concatenate((terrain, wind))

    def compress_wind(self, data):
        mean_x = data[-3,:,:]
        mean_z = data[-2,:,:]
        sig_xz = data[-1,:,:]

        idx = np.arange(0,self.size_z, self.box_size)
        if self.size_z%self.box_size != 0:
            idx = idx[:-1]
        pred = np.zeros((len(idx)*2)) # only consider 2 channels at the moment
        # wind
        for i in range(len(idx)):
            pred[i] = torch.mean(mean_x[:,idx[i]:idx[i] + self.box_size])
            pred[len(idx)+i] = torch.mean(mean_z[:,idx[i]:idx[i] + self.box_size])
        return pred

    def compress_terrain(self, data, position):
        rel_x = self.window_size + position[0] - int(position[0]) #to accound for the rounding that occured in window function
        rel_z = position[1]
        terrain = data[0,:,0]

        x = np.linspace(0,self.window_size_total,len(terrain))
        distances = []
        res = 100
        for i in range(len(terrain)*res):
            #distances.append(np.sqrt((rel_x-i)**2 + (rel_z-terrain[i])**2))
            distances.append(np.sqrt((rel_x-i/res)**2 + (rel_z-np.interp(i/res,x,terrain))**2))

        distance = np.min(distances)
        bearing = np.arctan2(np.argmin(distances)/res - rel_x, rel_z - np.interp(np.argmin(distances)/res,x,terrain))

        return [distance, bearing]

def load_tensor(path):
    name_list = os.listdir(path)
    name_list.sort()
    tensor_list = []
    for name in name_list:
        tensor = torch.load(path + name)
        tensor_list.append(tensor)
    return np.array(tensor_list)
