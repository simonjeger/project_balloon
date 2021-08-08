import numpy as np
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
    size_y = len(tensor[0][0])
    size_z = len(tensor[0][0][0])

    pos_x = int(position[0])
    pos_y = int(position[1])
    pos_z = int(position[2])

    pos_x = max(pos_x, 0)
    pos_x = min(pos_x, size_x-1)
    pos_y = max(pos_y, 0)
    pos_y = min(pos_y, size_y-1)
    pos_z = max(pos_z, 0)
    pos_z = min(pos_z, size_z-1)

    render_ratio = yaml_p['unit_xy']/yaml_p['unit_z']

    for dim in ['xz', 'yz', 'xy']: #xz, yz, xy
        if dim == 'xz':
            terrain = tensor[0,:,pos_y,0]
            dir_x = tensor[-4,:,pos_y,:]
            dir_y = tensor[-2,:,pos_y,:]
            sig = tensor[-1,:,pos_y,:]
        if dim == 'yz':
            terrain = tensor[0,pos_x,:,0]
            dir_x = tensor[-3,pos_x,:,:]
            dir_y = tensor[-2,pos_x,:,:]
            sig = tensor[-1,pos_x,:,:]
        if dim == 'xy':
            terrain = tensor[0,:,:,0]
            dir_x = tensor[-4,:,:,pos_z]
            dir_y = tensor[-3,:,:,pos_z]
            sig = tensor[-1,:,:,pos_z]

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
            dpi = 50
            ax.set_aspect(1)

            # plot terrain
            ax.fill_between(np.linspace(0,local_size_x,len(terrain)),terrain, color=c_terrain)

        else:
            # set parameters
            dpi = 70
            ax.set_aspect(1)

            """
            #TypeError: Shapes of x (216, 252) and z (433, 505) do not match
            dir_x = dir_x[1::2,1::2]
            dir_y = dir_y[1::2,1::2]
            local_size_x = int(local_size_x/2)
            local_size_y = int(local_size_y/2)
            """

            # plot contour lines
            x = np.arange(0, local_size_x, 1)
            y = np.arange(0, local_size_y, 1)

            x = np.linspace(0, local_size_x+1, local_size_x)
            y = np.linspace(0, local_size_y+1, local_size_y)

            X, Y = np.meshgrid(x, y)
            ax.contourf(X,Y,terrain.T, cmap=c_contour, extend=(0,local_size_x,0,local_size_y))
            ax.quiver(X, Y, dir_x, dir_y, scale=yaml_p['unit_xy']/5, headwidth=2.5, width=0.005)

        # save figure
        plt.savefig('render/wind_map_' + dim + '.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close()
