import numpy as np
import torch
from random import gauss
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cv2

def generate_wind_map(size_x, size_z, num, train_or_test):
    for n in range(num):
        # generate mean
        mean_x = np.ones(shape=(size_x,size_z))*0
        mean_z = np.ones(shape=(size_x,size_z))*0

        seed_x = np.random.uniform(low=-5, high=5)
        seed_z = np.random.uniform(low=-1, high=1)
        m = 10
        for i in range(m):
            rand_x = int(size_x - size_x/m*i - 1)
            rand_z = int(size_z - size_z/m*i - 1)
            mean_x[:, rand_z] = gauss(seed_x,5)
            mean_z[rand_x, :] = gauss(seed_z,1)
            mean_x = gaussian_filter(mean_x, sigma = 1)
            mean_z = gaussian_filter(mean_z, sigma = 5)

        mean_x /= np.max(abs(mean_x))
        mean_z /= np.max(abs(mean_z))

        # generate epistemic uncertainty
        #epi_x = np.random.rand(size_x,size_z)
        #epi_z = np.random.rand(size_x,size_z)
        #epi_x = gaussian_filter(epi_x, sigma = 10)
        #epi_z = gaussian_filter(epi_z, sigma = 10)
        epi_x = np.ones(shape=(size_x, size_z))*0
        epi_z = np.ones(shape=(size_x, size_z))*0

        # generate aleatoric uncertainty
        ale_x = np.ones(shape=(size_x, size_z))*0
        ale_z = np.ones(shape=(size_x, size_z))*0

        #tensor = np.dstack((mean_x, epi_x, ale_x, mean_z, epi_z, ale_z))
        #tensor = np.dstack((mean_x, mean_z, epi_x + ale_x + epi_z + ale_z))
        tensor = np.dstack((mean_x, mean_z))
        torch.save(tensor, 'data/' + train_or_test + '/tensor/wind_map' + str(n).zfill(5) + '.pt')

        print('generated ' + str(n+1) + ' of ' + str(num) + ' windmaps')
