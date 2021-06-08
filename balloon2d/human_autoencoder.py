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

        if yaml_p['type'] == 'regular':
            if yaml_p['autoencoder'] == 'HAE_avg':
                self.bottleneck_wind = int(self.size_z/self.box_size)*1 + 1 #because we mainly look at wind in x direction (and need to pass absolute hight)
            elif yaml_p['autoencoder'] == 'HAE_ext':
                self.bottleneck_wind = int(self.size_z/self.box_size)*1*self.window_size_total + 1 #because we mainly look at wind in x direction (and need to pass absolute hight)
            elif yaml_p['autoencoder'] == 'HAE_patch':
                self.bottleneck_wind = 2*1 + 2*1 #because we mainly look at wind in x direction
            else:
                print('ERROR: please choose one of the available HAE')

        elif yaml_p['type'] == 'squished':
            if yaml_p['autoencoder'] == 'HAE_avg':
                self.bottleneck_wind = int(self.size_z/self.box_size)*1 + 1 + 1 #because we mainly look at wind in x direction (and need to pass absolute hight)
            elif yaml_p['autoencoder'] == 'HAE_ext':
                self.bottleneck_wind = int(self.size_z/self.box_size)*1*self.window_size_total + 1 + 1 #because we mainly look at wind in x direction (and need to pass absolute hight)
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

        start_x = int(np.clip(center,0,self.size_x-1))
        end_x = int(start_x + self.window_size_total)

        window = data_padded[:,start_x:end_x,:]
        window = torch.tensor(window)
        return window

    def window_squished(self, data, center, ceiling):
        res = self.size_z
        data_squished = np.zeros((len(data),self.size_x,res))
        for i in range(self.size_x):
            bottom = data[0,i,0]
            top = ceiling

            x_old = np.arange(0,self.size_z,1)
            x_new = np.linspace(bottom,top,res)
            data_squished[0,:,:] = data[0,:,:] #terrain stays the same

            for j in range(1,len(data)):
                data_squished[j,i,:] = np.interp(x_new,x_old,data[j,i,:])

        data_padded = np.zeros((len(data_squished),self.size_x+2*self.window_size,res))
        data_padded[:,self.window_size:-self.window_size,:] = data_squished

        for i in range(self.window_size):
            data_padded[:,i,:] = data_padded[:,self.window_size,:]
            data_padded[:,-(i+1),:] = data_padded[:,-(self.window_size+1),:]

        start_x = int(np.clip(center,0,self.size_x-1))
        end_x = int(start_x + self.window_size_total)

        window = data_padded[:,start_x:end_x,:]
        window = torch.tensor(window)
        return window

    def compress(self, data, position, ceiling):
        if yaml_p['type'] == 'regular':
            window = self.window(data, position[0])
            if yaml_p['autoencoder'] == 'HAE_avg':
                wind = self.compress_wind_avg(window,position)
            elif yaml_p['autoencoder'] == 'HAE_ext':
                wind = self.compress_wind_ext(window,position)
            elif yaml_p['autoencoder'] == 'HAE_patch':
                wind = self.compress_wind_patch(window,position)

        elif yaml_p['type'] == 'squished':
            window = self.window_squished(data, position[0], ceiling)
            if yaml_p['autoencoder'] == 'HAE_avg':
                wind = self.compress_wind_avg_squished(window, position, ceiling)
            elif yaml_p['autoencoder'] == 'HAE_ext':
                wind = self.compress_wind_ext_squished(window, position, ceiling)
        return wind

    def compress_wind_avg(self, data, position):
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
        pred = np.zeros((len(idx)*1) + 1) # two different wind directions

        # wind
        for i in range(len(idx)):
            with warnings.catch_warnings(): #I expect to see RuntimeWarnings in this block
                warnings.simplefilter("ignore", category=RuntimeWarning)

                pred[i] = np.nanmean(mean_x[:,idx[i]:idx[i] + self.box_size])
                #pred[len(idx)+i] = torch.mean(mean_z[:,idx[i]:idx[i] + self.box_size])

        pred[-1] = position[1]

        pred = torch.tensor(np.nan_to_num(pred,0))

        return pred

    def compress_wind_ext(self, data, position):
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

        pred_x = np.zeros((len(idx),self.window_size_total))

        # wind
        for i in range(len(idx)):
            for j in range(self.window_size_total):
                with warnings.catch_warnings(): #I expect to see RuntimeWarnings in this block
                    warnings.simplefilter("ignore", category=RuntimeWarning)

                    pred_x[i,j] = np.nanmean(mean_x[j,idx[i]:idx[i] + self.box_size])

        pred = np.concatenate((pred_x.flatten(), [position[1]]))
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

        values_x = data[-3,loc_x,:]
        winner_min_x = np.argwhere(values_x == min(values_x)).flatten()
        dist_min_x = winner_min_x[np.argmin(abs(winner_min_x - loc_z))] - loc_z

        winner_max_x = np.argwhere(values_x == max(values_x)).flatten()
        dist_max_x = winner_max_x[np.argmin(abs(winner_max_x - loc_z))] - loc_z

        pred = np.array([idx_bottom_x - loc_z, idx_top_x - loc_z, dist_min_x, dist_max_x])

        return pred

    def compress_wind_avg_squished(self, data, position, ceiling):
        mean_x = data[-3,:,:]
        mean_z = data[-2,:,:]
        sig_xz = data[-1,:,:]

        idx = np.arange(0,self.size_z, self.box_size)
        if self.size_z%self.box_size != 0:
            idx = idx[:-1]
        pred = np.zeros((len(idx)*1) + 1 + 1) # two different wind directions

        # wind
        for i in range(len(idx)):
            with warnings.catch_warnings(): #I expect to see RuntimeWarnings in this block
                warnings.simplefilter("ignore", category=RuntimeWarning)

                pred[i] = np.nanmean(mean_x[:,idx[i]:idx[i] + self.box_size])
                #pred[len(idx)+i] = torch.mean(mean_z[:,idx[i]:idx[i] + self.box_size])

        pos_x = np.clip(int(position[0]),0,self.size_x - 1)
        rel_pos = torch.tensor([(position[1]-data[0,self.window_size,0]) / (ceiling - data[0,self.window_size,0])])
        size = (ceiling - data[0,self.window_size,0])/self.size_z

        pred[-2] = rel_pos
        pred[-1] = size

        pred = torch.tensor(np.nan_to_num(pred,0))

        return pred

    def compress_wind_ext_squished(self, data, position, ceiling):
        mean_x = data[-3,:,:]
        mean_z = data[-2,:,:]
        sig_xz = data[-1,:,:]

        idx = np.arange(0,self.size_z, self.box_size)
        if self.size_z%self.box_size != 0:
            idx = idx[:-1]
        pred_x = np.zeros((len(idx),self.window_size_total))

        # wind
        for i in range(len(idx)):
            for j in range(self.window_size_total):
                with warnings.catch_warnings(): #I expect to see RuntimeWarnings in this block
                    warnings.simplefilter("ignore", category=RuntimeWarning)

                    pred_x[i,j] = np.nanmean(mean_x[j,idx[i]:idx[i] + self.box_size])

        pos_x = np.clip(int(position[0]),0,self.size_x - 1)
        rel_pos = torch.tensor([(position[1]-data[0,self.window_size,0]) / (ceiling - data[0,self.window_size,0])])
        size = (ceiling - data[0,self.window_size,0])/self.size_z

        pred = np.concatenate((pred_x.flatten(), [rel_pos, size]))
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
