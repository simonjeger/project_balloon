import numpy as np
import torch
from random import gauss
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cv2
import random

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

def generate_world(size_x, size_y, size_z, num, train_or_test):
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
        torch.save(world, yaml_p['data_path'] + train_or_test + '/tensor/wind_map' + str(n).zfill(5) + '.pt')
        print('generated ' + str(n+1) + ' of ' + str(num) + ' worlds')

def generate_terrain(size_x, size_y, size_z):
    # generate mean & uncertainty
    terrain = np.zeros(shape=(size_x,size_y))

    min_sky = size_z/10*2
    m = int(size_x)
    magnitude = 75
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
    """
    terrain *= 0
    """
    return terrain

def generate_wind(size_x, size_y, size_z, terrain):
    # generate mean & uncertainty
    mean_x = np.ones(shape=(size_x,size_y,size_z))*0
    mean_y = np.ones(shape=(size_x,size_y,size_z))*0
    mean_z = np.ones(shape=(size_x,size_y,size_z))*0
    sig = np.ones(shape=(size_x,size_y,size_z))*0.1

    magnitude = 10
    smear_xy = max(int(size_x),1)
    smear_z = max(int(size_z/10),1)
    m = int(size_z)
    sign_x = random.choice([-1,1])
    sign_y = random.choice([-1,1])
    for i in range(m):
        pos_x = random.randint(0,size_x-1)
        pos_y = random.randint(0,size_y-1)
        pos_z = random.randint(0,size_z-1)
        if random.uniform(0, 1) < pos_z/size_z:
            seed_x = 1
            seed_y = 1
        else:
            seed_x = -1
            seed_y = -1

        seed_x *= sign_x
        seed_y *= sign_y
        mean_x[pos_x-smear_xy:pos_x+smear_xy, pos_y-smear_xy:pos_y+smear_xy, pos_z-smear_z:pos_z+smear_z] = gauss(magnitude*seed_x,10)
        mean_y[pos_x-smear_xy:pos_x+smear_xy, pos_y-smear_xy:pos_y+smear_xy, pos_z-smear_z:pos_z+smear_z] = gauss(magnitude*seed_y,10)
        mean_z[pos_x-smear_xy:pos_x+smear_xy, pos_y-smear_xy:pos_y+smear_xy, pos_z-smear_xy:pos_z+smear_xy] = gauss(0,10)
        sig[pos_x, pos_y, pos_z] = abs(gauss(1,1))
    mean_x = gaussian_filter(mean_x, sigma = 20)
    mean_y = gaussian_filter(mean_y, sigma = 20)
    mean_z = gaussian_filter(mean_z, sigma = 20)
    sig = gaussian_filter(sig, sigma = 20)

    # for homogeneous field
    """
    mean_x *= 0
    mean_x += 0.33
    #mean_x[:,5] += 0.33
    mean_y *= 0
    mean_y += 0.33
    mean_z *= 0
    sig *= 0
    sig += 0.5
    """

    return [mean_x, mean_y, mean_z, sig]
