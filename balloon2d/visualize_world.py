from preprocess_wind import squish, unsquish

import numpy as np
import scipy
import torch
from random import gauss
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import os
import cv2

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

def visualize_world(tensor, position, ceiling):
    size_x = len(tensor[0])
    size_z = len(tensor[0][0])

    pos_x = int(position[0])
    pos_z = int(position[1])

    pos_x = max(pos_x, 0)
    pos_x = min(pos_x, size_x-1)
    pos_z = max(pos_z, 0)
    pos_z = min(pos_z, size_z-1)

    render_ratio = yaml_p['unit_x']/yaml_p['unit_z']

    """
    # interpolate wind in squished frame
    tensor_squished = squish(tensor,ceiling)

    x = len(tensor_squished[0])
    y = len(tensor_squished[0][0])
    f_squished_0 = scipy.interpolate.interp2d(np.arange(x), np.arange(y), tensor_squished[0].T)
    f_squished_1 = scipy.interpolate.interp2d(np.arange(x), np.arange(y), tensor_squished[1].T)
    f_squished_2 = scipy.interpolate.interp2d(np.arange(x), np.arange(y), tensor_squished[2].T)
    f_squished_3 = scipy.interpolate.interp2d(np.arange(x), np.arange(y), tensor_squished[3].T)

    upsample_factor = render_ratio
    res = 1

    tensor_squished_0 = f_squished_0(np.linspace(0,x,int(x*upsample_factor*res)), np.linspace(0,y,int(y*res)))*res
    tensor_squished_1 = f_squished_1(np.linspace(0,x,int(x*upsample_factor*res)), np.linspace(0,y,int(y*res)))
    tensor_squished_2 = f_squished_2(np.linspace(0,x,int(x*upsample_factor*res)), np.linspace(0,y,int(y*res)))
    tensor_squished_3 = f_squished_3(np.linspace(0,x,int(x*upsample_factor*res)), np.linspace(0,y,int(y*res)))
    tensor_squished = np.array([tensor_squished_0.T, tensor_squished_1.T, tensor_squished_2.T, tensor_squished_3.T])
    tensor = unsquish(tensor_squished,ceiling*res)
    """


    for dim in ['xz']:
        if dim == 'xz':
            terrain = tensor[0,:,0]
            dir_x = tensor[-3,:,:]
            dir_y = tensor[-2,:,:]
            sig = tensor[-1,:,:]

        local_size_x = len(dir_x)
        local_size_y = len(dir_x[0])

        fig, ax = plt.subplots(frameon=False, figsize=(local_size_x, local_size_y))
        ax.set_axis_off()

        if dim != 'xy':
            cmap = sns.diverging_palette(145, 300, s=50, center="dark", as_cmap=True)
            #ax.imshow(dir_y.T, origin='lower', extent=[0, local_size_x, 0, local_size_y], cmap=cmap, alpha=1, vmin=-5, vmax=5, interpolation='bilinear')
            ax.imshow(dir_y.T, origin='lower', extent=[0, local_size_x, 0, local_size_y], cmap=cmap, alpha=1, vmin=-5, vmax=5)

            cmap = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
            #ax.imshow(dir_x.T, origin='lower', extent=[0, local_size_x, 0, local_size_y], cmap=cmap, alpha=0.7, vmin=-5, vmax=5, interpolation='bilinear')
            ax.imshow(dir_x.T, origin='lower', extent=[0, local_size_x, 0, local_size_y], cmap=cmap, alpha=0.7, vmin=-5, vmax=5)

        # draw terrain & coordinate system
        c_terrain = (161/255,135/255,93/255) #because plt uses values between 0 and 1
        c_contour = sns.cubehelix_palette(start=1.28, rot=0, dark=0.2, light=0.7, reverse=True, as_cmap=True)
        c_ticks = (242/255,242/255,242/255)

        if dim != 'xy':
            # set parameters
            dpi = 10
            ax.set_aspect(1)

            # plot terrain
            ax.fill_between(np.linspace(0,local_size_x,len(terrain)),terrain, color=c_terrain)

        # save figure
        plt.savefig('render/wind_map_' + dim + '.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close()
