import os
import glob

import numpy as np
import warnings
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split

from preprocess_wind import squish

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
        name_list = os.listdir(yaml_p['data_path'] + 'test/tensor/')
        tensor = torch.load(yaml_p['data_path'] + 'test/tensor/' + name_list[0])
        self.size_c = len(tensor)
        self.size_x = len(tensor[0])
        self.size_y = len(tensor[0][0])
        self.size_z = len(tensor[0][0][0])

        self.window_size = yaml_p['window_size']
        self.window_size_total = 2*self.window_size + 1
        self.box_size = int(self.size_z/yaml_p['bottleneck'])

        if yaml_p['autoencoder'] == 'HAE_avg':
            self.bottleneck_wind = int(self.size_z/self.box_size)*2 #because we mainly look at wind in x and y direction
        elif yaml_p['autoencoder'] == 'HAE_ext':
            self.bottleneck_wind = int(self.size_z/self.box_size)*2*self.window_size_total**2
        elif yaml_p['autoencoder'] == 'HAE_bidir':
            self.bottleneck_wind = 2*4
        elif yaml_p['autoencoder'] == 'HAE_special':
            self.window_size = 34
            self.window_size_total = 70
            self.bottleneck_wind = (1+2*4)*2
        else:
            print('ERROR: please choose one of the available HAE')

        self.bottleneck = self.bottleneck_wind

    def window(self, data, position):
        window = np.zeros((len(data),self.window_size_total,self.size_z))
        data_padded = np.zeros((len(data),self.size_x+2*self.window_size,self.size_y+2*self.window_size,self.size_z))
        if self.window_size == 0:
            data_padded = data
        else:
            data_padded[:,self.window_size:-self.window_size,self.window_size:-self.window_size,:] = data

        for i in range(self.window_size):
            data_padded[:,i,:,:] = data_padded[:,self.window_size,:,:]
            data_padded[:,:,i,:] = data_padded[:,:,self.window_size,:]
            data_padded[:,-(i+1),:,:] = data_padded[:,-(self.window_size+1),:,:]
            data_padded[:,:,-(i+1),:] = data_padded[:,:,-(self.window_size+1),:]

        start_x = int(np.clip(position[0],0,self.size_x-1))
        start_y = int(np.clip(position[1],0,self.size_y-1))
        end_x = int(position[0] + self.window_size_total)
        end_y = int(position[1] + self.window_size_total)

        window = data_padded[:,start_x:end_x,start_y:end_y,:]
        window = torch.tensor(window)
        return window

    def window_squished(self, data, position, ceiling):
        data_squished = squish(data,ceiling)
        res = len(data_squished[0,0,0,:])

        data_padded = np.zeros((len(data_squished),self.size_x+2*self.window_size,self.size_y+2*self.window_size,self.size_z))
        if self.window_size == 0:
            data_padded = data_squished
        else:
            data_padded[:,self.window_size:-self.window_size,self.window_size:-self.window_size,:] = data_squished

        for i in range(self.window_size):
            data_padded[:,i,:,:] = data_padded[:,self.window_size,:,:]
            data_padded[:,:,i,:] = data_padded[:,:,self.window_size,:]
            data_padded[:,-(i+1),:,:] = data_padded[:,-(self.window_size+1),:,:]
            data_padded[:,:,-(i+1),:] = data_padded[:,:,-(self.window_size+1),:]

        start_x = int(np.clip(position[0],0,self.size_x-1))
        start_y = int(np.clip(position[1],0,self.size_y-1))
        end_x = int(start_x + self.window_size_total)
        end_y = int(start_y + self.window_size_total)

        window = data_padded[:,start_x:end_x,start_y:end_y,:]
        window = torch.tensor(window)
        return window

    def compress(self, data, position, ceiling):
        window = self.window_squished(data, position, ceiling)
        if yaml_p['autoencoder'] == 'HAE_avg':
            wind = self.compress_wind_avg_squished(window, position, ceiling)
        elif yaml_p['autoencoder'] == 'HAE_ext':
            wind = self.compress_wind_ext_squished(window, position, ceiling)
        elif yaml_p['autoencoder'] == 'HAE_bidir':
            wind = self.compress_wind_bidir_squished(window, position, ceiling)
        elif yaml_p['autoencoder'] == 'HAE_special':
            wind = self.compress_wind_special_squished(window, position, ceiling)
        return wind

    def compress_est(self, data, position, ceiling):
        window = self.window(data, position)
        if yaml_p['autoencoder'] == 'HAE_avg':
            wind = self.compress_wind_avg_squished(window, position, ceiling)
        elif yaml_p['autoencoder'] == 'HAE_ext':
            wind = self.compress_wind_ext_squished(window, position, ceiling)
        elif yaml_p['autoencoder'] == 'HAE_bidir':
            wind = self.compress_wind_bidir_squished(window, position, ceiling)
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

    def compress_wind_ext(self, data, position):
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

        pred_x = np.zeros((len(idx),self.window_size_total, self.window_size_total))
        pred_y = np.zeros((len(idx),self.window_size_total, self.window_size_total))

        # wind
        for i in range(len(idx)):
            for j in range(self.window_size_total):
                for k in range(self.window_size_total):
                    with warnings.catch_warnings(): #I expect to see RuntimeWarnings in this block
                        warnings.simplefilter("ignore", category=RuntimeWarning)

                        pred_x[i,j,k] = np.nanmean(mean_x[j,k,idx[i]:idx[i] + self.box_size])
                        pred_y[i,j,k] = np.nanmean(mean_x[j,k,idx[i]:idx[i] + self.box_size])

        pred = np.concatenate((pred_x.flatten(), pred_y.flatten()))
        pred = torch.tensor(np.nan_to_num(pred,0))

        return pred

    def compress_wind_avg_squished(self, data, position,ceiling):
        mean_x = data[-4,:,:]
        mean_y = data[-3,:,:]
        mean_z = data[-2,:,:]
        sig_xz = data[-1,:,:]

        idx = np.arange(0, self.size_z, self.box_size)
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

    def compress_wind_ext_squished(self, data, position,ceiling):
        mean_x = data[-4,:,:]
        mean_y = data[-3,:,:]
        mean_z = data[-2,:,:]
        sig_xz = data[-1,:,:]

        idx = np.arange(0,self.size_z, self.box_size)
        if self.size_z%self.box_size != 0:
            idx = idx[:-1]

        pred_x = np.zeros((len(idx),self.window_size_total, self.window_size_total))
        pred_y = np.zeros((len(idx),self.window_size_total, self.window_size_total))

        # wind
        for i in range(len(idx)):
            for j in range(self.window_size_total):
                for k in range(self.window_size_total):
                    with warnings.catch_warnings(): #I expect to see RuntimeWarnings in this block
                        warnings.simplefilter("ignore", category=RuntimeWarning)

                        pred_x[i,j,k] = np.nanmean(mean_x[j,k,idx[i]:idx[i] + self.box_size])
                        pred_y[i,j,k] = np.nanmean(mean_y[j,k,idx[i]:idx[i] + self.box_size])

        pos_x = np.clip(int(position[0]),0,self.size_x - 1)
        pos_y = np.clip(int(position[1]),0,self.size_y - 1)

        rel_pos = torch.tensor([(position[2]-data[0,self.window_size,self.window_size,0]) / (ceiling - data[0,self.window_size,self.window_size,0])])
        size = (ceiling - data[0,self.window_size,self.window_size,0])/self.size_z

        pred = np.concatenate((pred_x.flatten(), pred_y.flatten()))
        pred = torch.tensor(np.nan_to_num(pred,0))

        return pred

    def compress_wind_bidir_squished(self, data, position,ceiling):
        apart = int(0.1/2*self.size_z)

        pred = np.zeros(8) # two different wind directions

        for d in ['u', 'v']:
            if d == 'u':
                wind = np.mean(np.array(data[-4]), axis=(0,1))
                idx = [0,4]
            if d == 'v':
                wind = np.mean(np.array(data[-3]), axis=(0,1))
                idx = [4,8]
            grad = abs(np.gradient(wind))
            b1 = np.argmax(grad)
            grad[b1-apart:b1+apart] *= 0
            b2 = np.argmax(grad)
            h1 = min(b1, b2)
            h2 = max(b1, b2)

            m1 = wind[int(h1/2)]
            m2 = wind[h1 + int((h2 - h1)/2)]

            pred[idx[0]:idx[1]] = [h1/self.size_z,h2/self.size_z,m1,m2]
        return pred

    def compress_wind_special_squished(self, data, position,ceiling):
        level = 2 #level = 0 would be only one central window
        len_level_0 = 2
        len_level_1 = 10
        len_level_2 = 70

        mean_x = data[-4,:,:]
        mean_y = data[-3,:,:]
        mean_z = data[-2,:,:]
        sig_xz = data[-1,:,:]

        idx = np.arange(0,self.size_z, self.box_size)
        if self.size_z%self.box_size != 0:
            idx = idx[:-1]

        pred_x = np.zeros((level*4 + 1, len(idx)))
        pred_y = np.zeros((level*4 + 1, len(idx)))

        # wind
        fig, ax = plt.subplots(1)

        for i in range(len(data[0])):
            for j in range(len(data[0][0])):
                for k in range(len(idx)):
                    if (len_level_2/2 - len_level_0/2 <= i < len_level_2/2 + len_level_0/2) & (len_level_2/2 - len_level_0/2 <= j < len_level_2/2 + len_level_0/2):
                        box_id = 0
                        color = 'red'
                    elif (len_level_2/2 - len_level_1/2 <= i < len_level_2/2) & (len_level_2/2 - len_level_1/2 <= j < len_level_2/2):
                        box_id = 1
                        color = 'green'
                    elif (len_level_2/2 <= i < len_level_2/2 + len_level_1/2) & (len_level_2/2 - len_level_1/2 <= j < len_level_2/2):
                        box_id = 2
                        color = 'blue'
                    elif (len_level_2/2 - len_level_1/2 <= i < len_level_2/2) & (len_level_2/2 <= j < len_level_2/2 + len_level_1/2):
                        box_id = 3
                        color = 'yellow'
                    elif (len_level_2/2 <= i < len_level_2/2 + len_level_1/2) & (len_level_2/2 <= j < len_level_2/2 + len_level_1/2):
                        box_id = 4
                        color = 'pink'
                    elif (0 <= i < len_level_2/2) & (0 <= j < len_level_2/2):
                        box_id = 5
                        color = 'black'
                    elif (len_level_2/2 <= i < len_level_2) & (0 <= j < len_level_2/2):
                        box_id = 6
                        color = 'grey'
                    elif (0 <= i < len_level_2/2) & (len_level_2/2 <= j < len_level_2):
                        box_id = 7
                        color = 'orange'
                    elif (len_level_2/2 <= i < len_level_2) & (len_level_2/2 <= j < len_level_2):
                        box_id = 8
                        color = 'salmon'
                    else:
                        print("ERROR: couldn't match that to anything in the HAE_special")

                    with warnings.catch_warnings(): #I expect to see RuntimeWarnings in this block
                        warnings.simplefilter("ignore", category=RuntimeWarning)

                        ax.scatter(i,j,color=color)

                        pred_x[box_id,k] = np.nanmean(mean_x[i,j,idx[k]:idx[k] + self.box_size])
                        pred_y[box_id,k] = np.nanmean(mean_y[i,j,idx[k]:idx[k] + self.box_size])

        plt.savefig('debug_HAE_special.png')
        plt.close()

        pos_x = np.clip(int(position[0]),0,self.size_x - 1)
        pos_y = np.clip(int(position[1]),0,self.size_y - 1)

        rel_pos = torch.tensor([(position[2]-data[0,self.window_size,self.window_size,0]) / (ceiling - data[0,self.window_size,self.window_size,0])])
        size = (ceiling - data[0,self.window_size,self.window_size,0])/self.size_z

        pred = np.concatenate((pred_x.flatten(), pred_y.flatten()))
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
