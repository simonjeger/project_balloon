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

        if yaml_p['type'] == 'regular':
            if yaml_p['autoencoder'] == 'HAE_avg':
                self.bottleneck_wind = int(self.size_z/self.box_size)*2 + 1 #because wind in x and y direction (and need to pass absolute hight)
            elif yaml_p['autoencoder'] == 'HAE_ext':
                self.bottleneck_wind = int(self.size_z/self.box_size)*2 + 1 #because we mainly look at wind in x direction (and need to pass absolute hight)
            elif yaml_p['autoencoder'] == 'HAE_patch':
                self.bottleneck_wind = 2*2 + 2*2#because we mainly look at wind in x direction
            else:
                print('ERROR: please choose one of the available HAE')

        elif yaml_p['type'] == 'squished':
            if yaml_p['autoencoder'] == 'HAE_avg':
                self.bottleneck_wind = int(self.size_z/self.box_size)*2 + 1 + 1 #because we mainly look at wind in x direction
            elif yaml_p['autoencoder'] == 'HAE_ext':
                self.bottleneck_wind = int(self.size_z/self.box_size)*2 + 1 + 1 #because we mainly look at wind in x direction (and need to pass absolute hight)
            else:
                print('ERROR: please choose one of the available HAE')

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

    def window_squished(self, data, position, ceiling):
        res = self.size_z
        data_squished = np.zeros((len(data),self.size_x,self.size_y,res))
        for i in range(self.size_x):
            for j in range(self.size_y):
                bottom = data[0,i,j,0]
                top = ceiling[i,j]

                x_old = np.arange(0,self.size_z,1)
                x_new = np.linspace(bottom,top,res)
                data_squished[0,:,:,:] = data[0,:,:,:] #terrain stays the same

                for k in range(1,len(data)):
                    data_squished[k,i,j,:] = np.interp(x_new,x_old,data[k,i,j,:])

        data_padded = np.zeros((len(data_squished),self.size_x+2*self.window_size,self.size_y+2*self.window_size,self.size_z))

        data_padded[:,self.window_size:-self.window_size,self.window_size:-self.window_size,:] = data_squished

        for i in range(self.window_size):
            data_padded[:,i,:,:] = data_padded[:,self.window_size,:,:]
            data_padded[:,:,i,:] = data_padded[:,:,self.window_size,:]
            data_padded[:,-(i+1),:,:] = data_padded[:,-(self.window_size+1),:,:]
            data_padded[:,:,-(i+1),:] = data_padded[:,:,-(self.window_size+1),:]

        start_x = int(position[0])
        start_y = int(position[1])
        end_x = int(start_x + self.window_size_total)
        end_y = int(start_y + self.window_size_total)

        window = data_padded[:,start_x:end_x,start_y:end_y,:]
        window = torch.tensor(window)
        return window

    def compress(self, data, position, ceiling):
        if yaml_p['type'] == 'regular':
            window = self.window(data, position)
            if yaml_p['autoencoder'] == 'HAE_avg':
                wind = self.compress_wind_avg(window,position)
            elif yaml_p['autoencoder'] == 'HAE_ext':
                wind = self.compress_wind_ext(window,position)
            elif yaml_p['autoencoder'] == 'HAE_patch':
                wind = self.compress_wind_patch(window,position)

        elif yaml_p['type'] == 'squished':
            window = self.window_squished(data, position, ceiling)
            if yaml_p['autoencoder'] == 'HAE_avg':
                wind = self.compress_wind_avg_squished(window, position, ceiling)
            elif yaml_p['autoencoder'] == 'HAE_ext':
                wind = self.compress_wind_ext_squished(window, position, ceiling)
        return wind

    def compress_wind_avg(self, data, position):
        # get rid of wind data that's below the terrain
        loc_x = len(data[0,:,0,0])
        loc_y = len(data[0,0,:,0])
        loc_z = len(data[0,0,0,:])

        corrected_data = data
        for i in range(loc_x):
            for j in range(loc_y):
                k = int(data[0,i,j,0])
                corrected_data[1:,i,j,0:k] = 0

        corrected_data = corrected_data.detach()

        mean_x = corrected_data[-4,:,:]
        mean_y = corrected_data[-3,:,:]
        mean_z = corrected_data[-2,:,:]
        sig_xz = corrected_data[-1,:,:]

        idx = np.arange(0,self.size_z, self.box_size)
        if self.size_z%self.box_size != 0:
            idx = idx[:-1]
        pred = np.zeros((len(idx)*2) + 1) # two different wind directions

        # wind
        for i in range(len(idx)):
            with warnings.catch_warnings(): #I expect to see RuntimeWarnings in this block
                warnings.simplefilter("ignore", category=RuntimeWarning)

                pred[0*len(idx)+i] = np.nanmean(mean_x[:,:,idx[i]:idx[i] + self.box_size])
                pred[1*len(idx)+i] = np.nanmean(mean_y[:,:,idx[i]:idx[i] + self.box_size])
                #pred[2*len(idx)+i] = torch.mean(mean_z[:,:,idx[i]:idx[i] + self.box_size])

        pred[-1] = position[2]

        pred = torch.tensor(np.nan_to_num(pred,0))
        return pred

    def compress_wind_ext(self, data, position):
        N = 5
        if yaml_p['window_size'] != 6:
            print('ERROR: compress_wind_ext requires window_size = 6')

        # get rid of wind data that's below the terrain
        loc_x = len(data[0,:,0,0])
        loc_y = len(data[0,0,:,0])
        loc_z = len(data[0,0,0,:])

        corrected_data = data
        for i in range(loc_x):
            for j in range(loc_y):
                k = int(data[0,i,j,0])
                corrected_data[1:,i,j,0:k] = 0

        corrected_data = corrected_data.detach()

        mean_x = corrected_data[-4,:,:]
        mean_y = corrected_data[-3,:,:]
        mean_z = corrected_data[-2,:,:]
        sig_xz = corrected_data[-1,:,:]

        idx_x = [0,4,6,7,9,13]
        idx_z = np.arange(0,self.size_z, self.box_size)
        if self.size_z%self.box_size != 0:
            idx_z = idx_z[:-1]
        pred = np.zeros((len(idx_z)*2) + 1) # two different wind directions

        # wind
        for j in range(N):
            for i in range(len(idx_z)):
                with warnings.catch_warnings(): #I expect to see RuntimeWarnings in this block
                    warnings.simplefilter("ignore", category=RuntimeWarning)

                    pred[0*len(idx_z)+i] = np.nanmean(mean_x[:,:,idx_z[i]:idx_z[i] + self.box_size])
                    pred[1*len(idx_z)+i] = np.nanmean(mean_y[:,:,idx_z[i]:idx_z[i] + self.box_size])
                    #pred[2*len(idx_z)+i] = torch.mean(mean_z[:,:,idx_z[i]:idx_Z[i] + self.box_size])

        pred[-1] = position[2]

        pred = torch.tensor(np.nan_to_num(pred,0))
        return pred

    def compress_wind_patch(self, data, position):
        loc_x = int(self.window_size)
        loc_y = int(self.window_size)
        loc_z = int(np.clip(position[2],0,self.size_z-1))

        # top border / bottom border
        sign_loc_x = np.sign(data[-4,loc_x,loc_y,loc_z])
        sign_loc_y = np.sign(data[-3,loc_x,loc_y,loc_z])
        sign_wind_x = np.sign(data[-4,loc_x,loc_y,:])
        sign_wind_y = np.sign(data[-3,loc_x,loc_y,:])

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

        dist_border_y = np.zeros(self.size_z)
        for i in range(self.size_z):
            if sign_wind_y[i] != sign_loc_y:
                dist_border_y[i] = 1/(i - loc_z)
        idx_bottom_y = np.argmin(dist_border_y)
        idx_top_y = np.argmax(dist_border_y)
        if dist_border_y[idx_bottom_y] == 0:
            idx_bottom_y = 0
        if dist_border_y[idx_top_y] == 0:
            idx_top_y = self.size_z

        # location of min / max
        values_x = data[-4,loc_x,loc_y,:]
        winner_min_x = np.argwhere(values_x == min(values_x)).flatten()
        dist_min_x = winner_min_x[np.argmin(abs(winner_min_x - loc_z))] - loc_z

        winner_max_x = np.argwhere(values_x == max(values_x)).flatten()
        dist_max_x = winner_max_x[np.argmin(abs(winner_max_x - loc_z))] - loc_z

        values_y = data[-3,loc_x,loc_y,:]
        winner_min_y = np.argwhere(values_y == min(values_y)).flatten()
        dist_min_y = winner_min_y[np.argmin(abs(winner_min_y - loc_z))] - loc_z

        winner_max_y = np.argwhere(values_y == max(values_y)).flatten()
        dist_max_y = winner_max_y[np.argmin(abs(winner_max_y - loc_z))] - loc_z

        pred = np.array([idx_bottom_x - loc_z, idx_top_x - loc_z, idx_bottom_y - loc_z, idx_top_y - loc_z, dist_min_x, dist_max_x, dist_min_y, dist_max_y])
        return pred

    def compress_wind_squished(self, data, position,ceiling):
        mean_x = data[-4,:,:]
        mean_y = data[-3,:,:]
        mean_z = data[-2,:,:]
        sig_xz = data[-1,:,:]

        idx = np.arange(0,self.size_z, self.box_size)
        if self.size_z%self.box_size != 0:
            idx = idx[:-1]
        pred = np.zeros((len(idx)*2) + 1 + 1) # two different wind directions

        # wind
        for i in range(len(idx)):
            with warnings.catch_warnings(): #I expect to see RuntimeWarnings in this block
                warnings.simplefilter("ignore", category=RuntimeWarning)

                pred[0*len(idx)+i] = np.nanmean(mean_x[:,:,idx[i]:idx[i] + self.box_size])
                pred[1*len(idx)+i] = np.nanmean(mean_y[:,:,idx[i]:idx[i] + self.box_size])
                #pred[2*len(idx)+i] = torch.mean(mean_z[:,:,idx[i]:idx[i] + self.box_size])

        pos_x = np.clip(int(position[0]),0,self.size_x - 1)
        pos_y = np.clip(int(position[1]),0,self.size_y - 1)

        rel_pos = torch.tensor([(position[2]-data[0,self.window_size,self.window_size,0]) / (ceiling[pos_x,pos_y] - data[0,self.window_size,self.window_size,0])])
        size = (ceiling[pos_x,pos_y] - data[0,self.window_size,self.window_size,0])/self.size_z
        pred[-2] = rel_pos
        pred[-1] = size

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
