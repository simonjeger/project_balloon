import numpy as np
import torch
from random import gauss
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cv2

def generate_world(size_x, size_z, num, train_or_test):
    for n in range(num):
        terrain = generate_terrain(size_x)
        wind = generate_wind(size_x, size_z, terrain)

        world = np.zeros(shape=(size_x,size_z,1+3))

        for i in range(size_x):
            for j in range(size_z):
                world[i,0,0] = terrain[i]
                if j < np.floor(terrain[i]):
                    world[i,j,1:] = 0
                else:
                    world[i,j,1:] = wind[i,j,:]

        # stack and save
        torch.save(world, 'data/' + train_or_test + '/tensor/wind_map' + str(n).zfill(5) + '.pt')
        print('generated ' + str(n+1) + ' of ' + str(num) + ' worlds')

def generate_terrain(size_x):
    # generate mean & uncertainty
    terrain = np.ones(shape=(size_x,1))*0

    m = 3
    for i in range(m):
        pos_x = int(size_x/(2*m) + size_x/m*i)
        terrain[pos_x] = abs(gauss(1,10))
        terrain = gaussian_filter(terrain, sigma = 5)

    # for homogeneous field
    """
    terrain *= 0
    """
    return terrain

def generate_wind(size_x, size_z, terrain):
    # generate mean & uncertainty
    mean_x = np.ones(shape=(size_x,size_z))*0
    mean_z = np.ones(shape=(size_x,size_z))*0
    sig_xz = np.ones(shape=(size_x,size_z))*0.01

    m = 2
    seed_x = np.random.choice([-1, 1])
    for i in range(m):
        rand_x = int(size_x/(2*m) + size_x/m*i)
        rand_z = int(size_z/(2*m) + size_z/m*i)
        mean_x[:, rand_z] = gauss(2*seed_x*(-1)**i,1)
        mean_z[rand_x, :] = gauss(0,1)
        sig_xz[rand_x, rand_z] = abs(gauss(0.2,1))

    mean_x = gaussian_filter(mean_x, sigma = 2)
    mean_z = gaussian_filter(mean_z, sigma = 2)
    sig_xz = gaussian_filter(sig_xz, sigma = 5)

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

    mean_x = gaussian_filter(mean_x, sigma = 2)
    mean_z = gaussian_filter(mean_z, sigma = 2)
    sig_xz = gaussian_filter(sig_xz, sigma = 5)

    # for homogeneous field
    """
    mean_x *= 0
    mean_x += 0.33
    mean_z *= 0
    sig_xz *= 0
    """

    return np.dstack((mean_x, mean_z, sig_xz))
