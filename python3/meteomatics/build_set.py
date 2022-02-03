from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import pandas as pd
import torch
import netCDF4
from pathlib import Path
import os
import shutil
import imageio
import datetime

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

def build_set(num, train_or_test):
    Path(yaml_p['data_path']).mkdir(parents=True, exist_ok=True)
    Path(yaml_p['data_path'] + train_or_test).mkdir(parents=True, exist_ok=True)
    Path(yaml_p['data_path'] + train_or_test + '/tensor').mkdir(parents=True, exist_ok=True)

    seed_overall = np.random.randint(0,2**32 - 1)
    num = int(num/4) #because the dataset will be 4 times bigger through the mirroring process

    # read in first dataset for dimensions
    list = os.listdir(yaml_p['process_path'] + 'data_cosmo/tensor/')

    tensor = torch.load(yaml_p['process_path'] + 'data_cosmo/tensor/' + list[0])
    size_x = yaml_p['size_x']
    size_y = yaml_p['size_y']
    size_z = yaml_p['size_z']
    size_c = len(tensor)
    global_size_x = len(tensor[0])
    global_size_y = len(tensor[0][0])
    global_size_z = len(tensor[0][0][0])

    # generate center coordinates
    IDX_x = np.random.randint(0,global_size_x - size_x - 1,num*10) #just in case that some of them lie in the mountains
    IDX_y = np.random.randint(0,global_size_y - size_y - 1,num*10) #just in case that some of them lie in the mountains

    for h in range(len(list)):
        tensor = torch.load(yaml_p['process_path'] + 'data_cosmo/tensor/' + list[h])
        coord = torch.load(yaml_p['process_path'] + 'data_cosmo/coord/' + list[h])
        name_time = list[h][-13:-3]

        for o in range(4):
            if o == 0:
                tensor_rot = tensor[:,:,:,:]
            elif o == 1:
                tensor_rot = tensor[:,::-1,:,:]
                tensor_rot[-4,:,:,:] = -tensor_rot[-4,:,:,:]
            elif o == 2:
                tensor_rot = tensor[:,:,::-1,:]
                tensor_rot[-3,:,:,:] = -tensor_rot[-3,:,:,:]
            elif o == 3:
                tensor_rot = tensor[:,::-1,::-1,:]
                tensor_rot[-4:-2,:,:,:] = -tensor_rot[-4:-2,:,:,:]

            world = np.ones(shape=(1+4,size_x,size_y,size_z))*size_z #so that it makes the first "while" for sure
            flat = 3
            center_x = int(len(world[0])/2-flat/2)
            center_y = int(len(world[0][0])/2-flat/2)
            n = 0
            s = 0
            while s < num:
                idx_x = IDX_x[n]
                idx_y = IDX_y[n]
                n += 1

                world = tensor_rot[:,idx_x:idx_x+size_x, idx_y:idx_y+size_y,0:size_z]
                coord_center = coord[:,idx_x + int(size_x/2), idx_y + int(size_y/2)] #only save the center coordinate (lat,lon)

                if np.max(world[0][center_x:center_x+flat,center_y:center_y+flat,0]) < (1 - yaml_p['min_space'])*size_z: #only generate maps with enough space
                    # naming convention
                    digits = 4
                    name_lat = str(int(np.round(coord_center[0],digits)*10**digits)).zfill(digits+2)
                    name_lon = str(int(np.round(coord_center[1],digits)*10**digits)).zfill(digits+2)
                    name_lat = name_lat[0:2] + '.' + name_lat[2::]
                    name_lon = name_lon[0:2] + '.' + name_lon[2::]
                    name = name_lat + '_' + name_lon + '_' + str(o) + '_' + name_time + '.pt'
                    torch.save(world, yaml_p['data_path'] + train_or_test + '/tensor/' + name)

                    print('generated ' + str(o*num + s + 1) + ' of ' + str(num*4) + ' maps at ' + str(h).zfill(2) + ':00')
                    s += 1

build_set(1000, 'train')
build_set(100, 'test')
