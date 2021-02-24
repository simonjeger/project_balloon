import numpy as np
import torch
from random import gauss
import matplotlib.pyplot as plt
import cv2

def generate_wind_map(size_x, size_z, num):
    for n in range(num):
        # generate mean
        mean_x = np.ones(shape=(size_x,size_z))
        mean_z = np.ones(shape=(size_x,size_z))
        mean_x[0,:] = np.random.normal(1,0.5,size_z)
        mean_x[:,0] = np.random.normal(1,0.5,size_x)
        mean_z[0,:] = np.random.normal(0.2,0.1,size_z)
        mean_z[:,0] = np.random.normal(0.2,0.1,size_x)
        for i in range(size_x-1):
            for j in range(size_z-1):
                mean_x[i+1][j+1] = np.mean([mean_x[i][j], mean_x[i+1][j], mean_x[i][j+1]]) + gauss(0,0.5)
                mean_z[i+1][j+1] = np.mean([mean_z[i][j], mean_z[i+1][j], mean_z[i][j+1]]) + gauss(0,0.1)

        # generate epistemic uncertainty
        epi_x = np.random.rand(size_x,size_z)*0.02
        epi_z = np.random.rand(size_x,size_z)*0.01

        # generate aleatoric uncertainty
        ale_x = np.ones(shape=(size_x, size_z))*0.005
        ale_z = np.ones(shape=(size_x, size_z))*0.002

        # generate visual plot of wind_map
        sig_xz = epi_x + ale_x + epi_z + ale_z
        z,x = np.meshgrid(np.arange(0, size_z, 1),np.arange(0, size_x, 1))
        fig, ax = plt.subplots(frameon=False)
        ax.set_axis_off()
        ax.set_aspect(1)
        # generate quiver
        q = ax.quiver(x,z,mean_x,mean_z,sig_xz)
        # save figure
        plt.savefig('data/image/wind_map' + str(n) + '.png', dpi = 500, bbox_inches='tight')
        # rotate figure
        #img = cv2.imread('wind_map.png')
        #img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # crop image
        """border_x = 150
        border_y = 50
        #img = img[border_y:len(img)-border_y,border_x:len(img[0])-border_x]"""
        #cv2.imwrite('wind_map.png', img)

        tensor = [mean_x, epi_x, ale_x, mean_z, epi_z, ale_z]
        torch.save(tensor, 'data/tensor/wind_map_' + str(n) + '.pt')

        print('generated ' + str(n) + ' of ' + str(num) + ' windmaps')

generate_wind_map(100,30,100)
