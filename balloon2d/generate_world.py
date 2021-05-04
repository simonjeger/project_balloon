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

def generate_world(num, train_or_test):
    size_x = yaml_p['size_x']
    size_z = yaml_p['size_z']
    for n in range(num):
        terrain = generate_terrain(size_x, size_z)
        wind = generate_wind(size_x, size_z, terrain)

        world = np.zeros(shape=(1+3,size_x,size_z))

        for i in range(size_x):
            world[0,i,0] = terrain[i]

            for j in range(size_z):
                if j >= np.floor(terrain[i]):
                    world[-3,i,j] = wind[0][i,j]
                    world[-2,i,j] = wind[1][i,j]
                    world[-1,i,j] = wind[2][i,j]

        # save
        torch.save(world, yaml_p['data_path'] + train_or_test + '/tensor/wind_map' + str(n).zfill(5) + '.pt')
        print('generated ' + str(n+1) + ' of ' + str(num) + ' worlds')

def generate_terrain(size_x, size_z):
    # generate mean & uncertainty
    terrain = np.zeros(shape=(size_x,1))

    min_sky = size_z/10*2
    m = int(size_x)
    for i in range(m):
        #pos_x = int(size_x/(2*m) + size_x/m*i)
        pos_x = random.randint(0, size_x-1)
        terrain[pos_x] = gauss(size_z/7,size_z/5) #0,2

    terrain = gaussian_filter(terrain, sigma = 10)

    for i in range(len(terrain)):
        terrain[i] = min(terrain[i], size_z - min_sky)
        terrain[i] = max(terrain[i], 0)

    # for homogeneous field
    """
    terrain *= 0
    """
    return terrain

def generate_wind(size_x, size_z, terrain):
    # generate mean & uncertainty
    mean_x = np.ones(shape=(size_x,size_z))*0
    mean_z = np.ones(shape=(size_x,size_z))*0
    sig_xz = np.ones(shape=(size_x,size_z))*0.1

    """
    m = int(size_z/2)
    seed_x = np.random.choice([-1, 1])
    magnitude = 10 #used to be 2
    for i in range(m):
        rand_x = int(size_x/(2*m) + size_x/m*i)
        rand_z = int(size_z/(2*m) + size_z/m*i)
        mean_x[:, rand_z] = gauss(magnitude*seed_x*(-1)**int(i/(0.5*m)),5)
        mean_z[rand_x, :] = gauss(0,5)
        sig_xz[rand_x, rand_z] = abs(gauss(1,1))
    """

    magnitude = 50 #used to be 2
    smear = int(size_x/10)
    m = int(size_z)
    sign = random.choice([-1,1])
    for i in range(m):
        pos_x = random.randint(0,size_x-1)
        pos_z = random.randint(0,size_z-1)
        if random.uniform(0, 1) < pos_z/size_z:
            seed_x = 1
        else:
            seed_x = -1
        seed_x *= sign
        mean_x[pos_x-smear:pos_x+smear, pos_z] = gauss(magnitude*seed_x,5)
        mean_z[pos_x, pos_z-smear:pos_z+smear] = gauss(0,2)
        sig_xz[pos_x, pos_z] = abs(gauss(1,1))

    mean_x = gaussian_filter(mean_x, sigma = 10) #used to be 2 everywhere
    mean_z = gaussian_filter(mean_z, sigma = 10)
    sig_xz = gaussian_filter(sig_xz, sigma = 10)

    """
    # to be more realistic at the grenzschicht
    t = size_x
    for i in range(t):
        i = int(size_x/t*i + 1/2*size_x/t)
        j = int(terrain[i])
        if mean_x[i,j] > 0:
            mean_x[i,j] = 1
            mean_z[i,j] = terrain[i] - terrain[i-1]
        else:
            mean_x[i,j] = -1
            mean_z[i,j] = terrain[i-1] - terrain[i]
        sig_xz[i,j] = abs(gauss(1,1))

    mean_x = gaussian_filter(mean_x, sigma = 10)
    mean_z = gaussian_filter(mean_z, sigma = 10)
    sig_xz = gaussian_filter(sig_xz, sigma = 5)
    """

    # for homogeneous field
    """
    mean_x *= 0
    mean_x += 0.33
    #mean_x[:,5] += 0.33
    mean_z *= 0
    sig_xz *= 0
    sig_xz += 0.5
    """

    return [mean_x, mean_z, sig_xz]
