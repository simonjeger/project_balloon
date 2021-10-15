import torch
import numpy as np
from random import gauss
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cv2
import random
from pathlib import Path

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

def generate_world(num, n_t, train_or_test):
    Path(yaml_p['data_path']).mkdir(parents=True, exist_ok=True)
    Path(yaml_p['data_path'] + train_or_test).mkdir(parents=True, exist_ok=True)
    Path(yaml_p['data_path'] + train_or_test + '/tensor').mkdir(parents=True, exist_ok=True)

    size_x = yaml_p['size_x']
    size_y = yaml_p['size_y']
    size_z = yaml_p['size_z']

    for n in range(num):
        terrain = generate_terrain(size_x, size_y, size_z)
        wind = generate_wind(size_x, size_y, size_z, terrain)

        world = np.zeros(shape=(1+4,size_x,size_y,size_z))

        for i in range(size_x):
            for j in range(size_y):
                world[0,i,j,0] = terrain[i,j]

                for k in range(size_z):
                    if k >= np.floor(terrain[i,j]):
                        world[-4,i,j,k] = wind[0][i,j,k]
                        world[-3,i,j,k] = wind[1][i,j,k]
                        world[-2,i,j,k] = wind[2][i,j,k]
                        world[-1,i,j,k] = wind[3][i,j,k]

        # save
        for t in range(n_t):
            torch.save(world, yaml_p['data_path'] + train_or_test + '/tensor/wind_map' + str(n).zfill(5) + '_' + str(t).zfill(2) + '.pt')
        print('generated ' + str(n+1) + ' of ' + str(num) + ' worlds')

def generate_terrain(size_x, size_y, size_z):
    # generate mean & uncertainty
    terrain = np.zeros(shape=(size_x,size_y))

    min_sky = size_z/10*2
    m = int(size_x)
    magnitude = 100
    for i in range(m):
        pos_x = random.randint(0, size_x-1)
        pos_y = random.randint(0, size_y-1)
        terrain[pos_x, pos_y] = gauss(magnitude,size_z/5) #0,2

    terrain = gaussian_filter(terrain, sigma = 10)

    for i in range(len(terrain)):
        for j in range(len(terrain[0])):
            terrain[i,j] = min(terrain[i,j], size_z - min_sky)
            terrain[i,j] = max(terrain[i,j], 0)

    # for homogeneous field
    terrain *= 0
    return terrain

def generate_wind(size_x, size_y, size_z, terrain):
    # parameters
    m_abs = 5
    w_min = 0.2

    # generate mean & uncertainty
    mean_x = np.ones(shape=(size_x,size_y,size_z))*np.random.uniform(-m_abs, m_abs)
    mean_y = np.ones(shape=(size_x,size_y,size_z))*np.random.uniform(-m_abs, m_abs)
    mean_z = np.ones(shape=(size_x,size_y,size_z))*0
    sig = np.ones(shape=(size_x,size_y,size_z))*0

    m_x = np.random.uniform(-m_abs, m_abs)
    m_y = np.random.uniform(-m_abs, m_abs)
    h = -1
    w = 0
    while (h < 0) | (h + w >= size_z):
        w = np.random.uniform(w_min, 1 - w_min)
        h = np.random.uniform()
        w_i = int(w*size_z)
        h_i = int(h*size_z)

    mean_x[:,:,h_i:h_i+w_i] = -np.sign(mean_x[0,0,0])*abs(m_x)
    mean_y[:,:,h_i:h_i+w_i] = -np.sign(mean_y[0,0,0])*abs(m_y)

    return [mean_x, mean_y, mean_z, sig]

def generate_wind_indoor(size_x, size_y, size_z, terrain):
    pos = [0,0,0]
    angle = [0,0,0]

    
    mean_x = np.zeros(shape=(size_x,size_y,size_z))
    mean_y = np.zeros(shape=(size_x,size_y,size_z))
    mean_z = np.zeros(shape=(size_x,size_y,size_z))

generate_world(500, 7, 'train')
generate_world(500, 7, 'test')
