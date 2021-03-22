import numpy as np
import torch
from random import gauss
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cv2

def generate_wind_map(size_x, size_z, num, train_or_test):

    for n in range(num):
        # generate mean & uncertainty
        mean_x = np.ones(shape=(size_x,size_z))*0
        mean_z = np.ones(shape=(size_x,size_z))*0
        sig_xz = np.ones(shape=(size_x,size_z))*0.01

        m = 2
        # seed_x = np.random.choice([-1, 1])
        seed_x = 1
        for i in range(m):
            rand_x = int(size_x - size_x/m*i - 1)
            rand_z = int(size_z - size_z/m*i - 1)
            mean_x[:, rand_z] = gauss(seed_x*(-1)**i,0.01) #non-uniform
            #mean_x[:, rand_z] = gauss(seed_x,0.01) #uniform
            mean_z[rand_x, :] = gauss(0,0.01)
            sig_xz[rand_x, rand_z] = abs(gauss(0,0.001))

            mean_x = gaussian_filter(mean_x, sigma = 2)
            mean_z = gaussian_filter(mean_z, sigma = 5)
            sig_xz = gaussian_filter(sig_xz, sigma = 5)

        mean_x /= np.max(abs(mean_x))
        mean_z /= np.max(abs(mean_z))*5
        sig_xz /= np.max(abs(sig_xz))*10

        sig_xz *= 0 # deterministic case

        # generate uncertainty
        tensor = np.dstack((mean_x, mean_z, sig_xz))
        torch.save(tensor, 'data/' + train_or_test + '/tensor/wind_map' + str(n).zfill(5) + '.pt')

        print('generated ' + str(n+1) + ' of ' + str(num) + ' windmaps')
