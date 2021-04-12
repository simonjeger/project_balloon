import numpy as np
import torch
from random import gauss
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cv2
import random

def generate_world(size_x, size_z, num, train_or_test):
    for n in range(num):
        terrain = generate_terrain(size_x, size_z)
        wind = generate_wind(size_x, size_z, terrain)

        world = np.zeros(shape=(3+3,size_x,size_z))

        for i in range(size_x):
            world[0,i,0] = terrain[i]

            for j in range(size_z):
                distances = []
                for k in range(len(terrain)):
                    distances.append(np.sqrt((i-k)**2 + (j-terrain[k])**2))

                if j >= np.floor(terrain[i]):
                    world[1,i,j] = np.min(distances)
                    world[2,i,j] = np.arctan2(np.argmin(distances) - i,j - terrain[np.argmin(distances)])
                    world[-3,i,j] = wind[0][i,j]
                    world[-2,i,j] = wind[1][i,j]
                    world[-1,i,j] = wind[2][i,j]

        """
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(terrain)
        ax.set_aspect('equal')
        ax.set_xlim(0,size_x)
        ax.set_ylim(0,size_z)
        plt.savefig('terrain.png')
        plt.close()

        plt.imshow(world[1,:,:])
        plt.colorbar()
        plt.savefig('distance.png')
        plt.close()

        plt.imshow(world[2,:,:])
        plt.colorbar()
        plt.savefig('bearing.png')
        plt.close()
        """

        # save
        torch.save(world, 'data/' + train_or_test + '/tensor/wind_map' + str(n).zfill(5) + '.pt')
        print('generated ' + str(n+1) + ' of ' + str(num) + ' worlds')

def generate_terrain(size_x, size_z):
    # generate mean & uncertainty
    terrain = np.ones(shape=(size_x,1))*0

    min_sky = 2
    m = 10
    for i in range(m):
        #pos_x = int(size_x/(2*m) + size_x/m*i)
        pos_x = random.randint(0, size_x-1)
        terrain[pos_x] = abs(gauss(0,5))
        terrain = gaussian_filter(terrain, sigma = 1)

        for j in range(len(terrain)):
            terrain[j] = min(terrain[j], size_z - min_sky)

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
        sig_xz[rand_x, rand_z] = abs(gauss(1,1))

    mean_x = gaussian_filter(mean_x, sigma = 2)
    mean_z = gaussian_filter(mean_z, sigma = 2)
    sig_xz = gaussian_filter(sig_xz, sigma = 2)

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

    mean_x = gaussian_filter(mean_x, sigma = 2)
    mean_z = gaussian_filter(mean_z, sigma = 2)
    sig_xz = gaussian_filter(sig_xz, sigma = 2)

    # for homogeneous field
    """
    mean_x *= 0
    mean_x += 0.33
    mean_z *= 0
    sig_xz *= 0
    """

    return [mean_x, mean_z, sig_xz]
