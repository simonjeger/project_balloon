import os
import glob

import numpy as np
import warnings
import scipy
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
        self.size_y = len(tensor[0][0])
        self.size_z = len(tensor[0][0][0])

        self.window_size = yaml_p['window_size']
        self.window_size_total = 2*self.window_size + 1
        self.box_size = int(self.size_z/yaml_p['bottleneck'])

        self.bottleneck_wind = int(self.size_z/self.box_size)*2 #because wind in x and y direction
        self.bottleneck = self.bottleneck_wind

    def window(self, data, position):
        window = np.zeros((len(data),self.window_size_total,self.size_z))
        data_padded = np.zeros((len(data),self.size_x+2*self.window_size,self.size_y+2*self.window_size,self.size_z))

        data_padded[:,self.window_size:-self.window_size,self.window_size:-self.window_size,:] = data

        for i in range(self.window_size):
            data_padded[:,i,:,:] = data_padded[:,self.window_size,:,:]
            data_padded[:,:,i,:] = data_padded[:,:,self.window_size,:]
            data_padded[:,-(i+1),:,:] = data_padded[:,-(self.window_size+1),:,:]
            data_padded[:,:,-(i+1),:] = data_padded[:,:,-(self.window_size+1),:]

        start_x = int(position[0])
        start_y = int(position[1])
        end_x = int(position[0] + self.window_size_total)
        end_y = int(position[1] + self.window_size_total)

        window = data_padded[:,start_x:end_x,start_y:end_y,:]
        window = torch.tensor(window)
        return window

    def compress(self, data, position):
        window = self.window(data, position)
        wind = self.compress_wind(window)
        return wind

    def compress_wind(self, data):
        # get rid of wind data that's below the terrain
        loc_x = len(data[0,:,0,0])
        loc_y = len(data[0,0,:,0])
        loc_z = len(data[0,0,0,:])

        corrected_data = np.zeros((len(data), loc_x, loc_y, loc_z))
        for i in range(loc_x):
            for j in range(loc_y):
                for k in range(loc_z):
                    if k < data[0,i,j,0]:
                        corrected_data[1:,i,j,0:k] = np.nan
                    else:
                        corrected_data[:,i,j,k] = data[:,i,j,k]

        corrected_data = torch.tensor(corrected_data)
        corrected_data = corrected_data.detach()

        mean_x = corrected_data[-4,:,:]
        mean_y = corrected_data[-3,:,:]
        mean_z = corrected_data[-2,:,:]
        sig_xz = corrected_data[-1,:,:]

        idx = np.arange(0,self.size_z, self.box_size)
        if self.size_z%self.box_size != 0:
            idx = idx[:-1]
        pred = np.zeros((len(idx)*2)) # two different wind directions

        # wind
        for i in range(len(idx)):
            with warnings.catch_warnings(): #I expect to see RuntimeWarnings in this block
                warnings.simplefilter("ignore", category=RuntimeWarning)

                pred[0*len(idx)+i] = np.nanmean(mean_x[:,:,idx[i]:idx[i] + self.box_size])
                pred[1*len(idx)+i] = np.nanmean(mean_y[:,:,idx[i]:idx[i] + self.box_size])
                #pred[2*len(idx)+i] = torch.mean(mean_z[:,:,idx[i]:idx[i] + self.box_size])

        pred = torch.tensor(np.nan_to_num(pred,0))
        return pred

def load_tensor(path):
    name_list = os.listdir(path)
    name_list.sort()
    tensor_list = []
    for name in name_list:
        tensor = torch.load(path + name)
        tensor_list.append(tensor)
    return np.array(tensor_list)
