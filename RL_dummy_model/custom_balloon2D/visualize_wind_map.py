import numpy as np
import torch
from random import gauss
import matplotlib.pyplot as plt
import os
import cv2

def visualize_wind_map(train_or_test, tensor = 'empty'):

    if type(tensor) is not str:
        tensor_list = tensor

    else:
        name_list = os.listdir('data/' + train_or_test + '/tensor/')
        name_list.sort()
        tensor_list = []
        for name in name_list:
            tensor_list.append(torch.load('data/' + train_or_test + '/tensor/' + name))

    num = len(tensor_list)
    for n in range(num):
        mean_x = tensor_list[n][:,:,0]
        mean_z = tensor_list[n][:,:,1]
        sig_xz = tensor_list[n][:,:,2]
        size_x = len(mean_x)
        size_z = len(mean_x[0])

        z,x = np.meshgrid(np.arange(0, size_z, 1),np.arange(0, size_x, 1))
        fig, ax = plt.subplots(frameon=False)
        ax.set_axis_off()
        ax.set_aspect(1)

        # generate quiver
        q = ax.quiver(x,z,mean_x,mean_z,sig_xz)

        if type(tensor) is not str:
            plt.show()
            plt.close()

        else:
            # save figure
            plt.savefig('data/' + train_or_test + '/image/wind_map' + str(n).zfill(5) + '.png', dpi = 500, bbox_inches='tight', pad_inches=0)
            plt.close()

            # read in image with cv to then crop it
            img = cv2.imread('data/' + train_or_test + '/image/wind_map' + str(n).zfill(5) + '.png', cv2.IMREAD_UNCHANGED)
            border_x = int(0.9*size_x)
            border_y = int(0.4*size_z)
            img = img[border_y:len(img)-border_y,border_x:len(img[0])-border_x]

            cv2.imwrite('data/' + train_or_test + '/image/wind_map' + str(n).zfill(5) + '.png', img)

            print('visualized ' + str(n+1) + ' of ' + str(num) + ' windmaps')
