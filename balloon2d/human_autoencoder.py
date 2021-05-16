import os
import glob

import numpy as np
import warnings
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

        if yaml_p['autoencoder'] == 'HAE_avg':
            self.bottleneck_wind = int(self.size_z/self.box_size)*1 #because we mainly look at wind in x direction
        elif yaml_p['autoencoder'] == 'HAE_patch':
            self.bottleneck_wind = 2*1 #because we mainly look at wind in x direction
        else:
            print('ERROR: please choose one of the available HAE')
        self.bottleneck = self.bottleneck_wind

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
        if yaml_p['autoencoder'] == 'HAE_avg':
            wind = self.compress_wind_avg(window)
        elif yaml_p['autoencoder'] == 'HAE_patch':
            wind = self.compress_wind_patch(window,position)
        return wind

    def compress_wind_avg(self, data):
        # get rid of wind data that's below the terrain
        loc_x = len(data[0,:,0])
        loc_z = len(data[0,0,:])

        corrected_data = data
        for i in range(loc_x):
            k = int(data[0,i,0])
            corrected_data[1:,i,0:k] = 0

        corrected_data = corrected_data.detach()

        mean_x = corrected_data[-3,:,:]
        mean_z = corrected_data[-2,:,:]
        sig_xz = corrected_data[-1,:,:]

        idx = np.arange(0,self.size_z, self.box_size)
        if self.size_z%self.box_size != 0:
            idx = idx[:-1]
        pred = np.zeros((len(idx)*1)) # two different wind directions

        # wind
        for i in range(len(idx)):
            with warnings.catch_warnings(): #I expect to see RuntimeWarnings in this block
                warnings.simplefilter("ignore", category=RuntimeWarning)

                pred[i] = np.nanmean(mean_x[:,idx[i]:idx[i] + self.box_size])
                #pred[len(idx)+i] = torch.mean(mean_z[:,idx[i]:idx[i] + self.box_size])

        pred = torch.tensor(np.nan_to_num(pred,0))

        return pred

    def compress_wind_patch(self, data, position):
        loc_x = int(self.window_size)
        loc_z = int(np.clip(position[1],0,self.size_z-1))

        # top border / bottom border
        sign_loc_x = np.sign(data[-3,loc_x,loc_z])
        sign_wind_x = np.sign(data[-3,loc_x,:])

        dist_border_x = np.zeros(self.size_z)
        for i in range(self.size_z):
            if sign_wind_x[i] != sign_loc_x:
                dist_border_x[i] = 1/(i - loc_z)
        idx_bottom_x = np.argmin(dist_border_x)
        idx_top_x = np.argmax(dist_border_x)

        if dist_border_x[idx_bottom_x] == 0:
            idx_bottom_x = 0
        if dist_border_x[idx_top_x] == 0:
            idx_top_x = self.size_z

        closest_border = np.array([idx_bottom_x - loc_z, idx_top_x - loc_z])
        return closest_border

def load_tensor(path):
    name_list = os.listdir(path)
    name_list.sort()
    tensor_list = []
    for name in name_list:
        tensor = torch.load(path + name)
        tensor_list.append(tensor)
    return np.array(tensor_list)
