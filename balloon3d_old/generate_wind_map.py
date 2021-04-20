import numpy as np
import torch
from random import gauss
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cv2

def generate_wind_map(size_x, size_y, size_z, num, train_or_test):
    for n in range(num):
        # generate mean & uncertainty
        u = np.ones(shape=(size_x,size_y,size_z))*0
        v = np.ones(shape=(size_x,size_y,size_z))*0
        w = np.ones(shape=(size_x,size_y,size_z))*0
        turb = np.ones(shape=(size_x,size_y,size_z))*0.01

        m = 2
        seed_u = np.random.choice([-1, 1])
        seed_v = np.random.choice([-1, 1])
        for i in range(m):
            rand_x = int(size_x - size_x/m*i - 1)
            rand_y = int(size_y - size_y/m*i - 1)
            rand_z = int(size_z - size_z/m*i - 1)
            u[:,:,rand_z] = gauss(seed_u*(-1)**i,0.1)
            v[:,:,rand_z] = gauss(seed_v*(-1)**i,0.1)
            w[rand_x,rand_y,:] = gauss(0,1)
            turb[rand_x,rand_y,rand_z] = abs(gauss(0,1))
            u = gaussian_filter(u, sigma = 2)
            v = gaussian_filter(u, sigma = 2)
            w = gaussian_filter(w, sigma = 5)
            turb = gaussian_filter(turb, sigma = 5)

        u /= np.max(abs(u))
        v /= np.max(abs(v))*2
        w /= np.max(abs(w))*3
        turb /= np.max(abs(turb))*10

        # generate map
        tensor = np.stack((u, v, w, turb), axis=-1)
        torch.save(tensor, 'data/' + train_or_test + '/tensor/wind_map' + str(n).zfill(5) + '.pt')

        print('generated ' + str(n+1) + ' of ' + str(num) + ' windmaps')
