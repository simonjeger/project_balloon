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

def visualize_world(tensor, position):
    size_x = len(tensor[0])
    size_y = len(tensor[0][0])
    size_z = len(tensor[0][0][0])

    pos_x = int(position[0])
    pos_y = int(position[1])
    pos_z = int(position[2])

    pos_x = max(pos_x, 0)
    pos_x = min(pos_x, size_x)
    pos_y = max(pos_y, 0)
    pos_y = min(pos_y, size_y)
    pos_z = max(pos_z, 0)
    pos_z = min(pos_z, size_z)

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

        y,x = np.meshgrid(np.arange(0, local_size_y, 1),np.arange(0, local_size_x, 1))
        fig, ax = plt.subplots(frameon=False, figsize=(local_size_x, local_size_y))
        ax.set_axis_off()
        ax.set_aspect(1)

        if dim != 'xy':
            # standardise color map
            ceil = 10
            color_quiver = dir_x.copy()
            color_quiver = np.maximum(color_quiver, -ceil)
            color_quiver = np.minimum(color_quiver, ceil)
            color_quiver /= 2*ceil
            color_quiver += 0.5
            cm = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
            colors = cm(color_quiver).reshape(local_size_x*local_size_y,4)
            q = ax.quiver(x, y, dir_x, dir_y, color=colors, scale=50*yaml_p['unit'], headwidth=3, width=0.0015)

            c_terrain = (169/255,163/255,144/255) #because plt uses values between 0 and 1
            ax.fill_between(np.linspace(0,local_size_x,len(terrain)),terrain, color=c_terrain)
        else:
            # standardise color map
            ceil = 10
            color_quiver = np.arctan2(dir_y, dir_x)
            color_quiver /= 2*np.pi
            color_quiver += 0.5
            cm = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
            colors = cm(color_quiver).reshape(local_size_x*local_size_y,4)
            q = ax.quiver(x, y, dir_x, dir_y, color=colors, scale=50*yaml_p['unit'], headwidth=3, width=0.0015)

            #c_terrain = 'YlOrBr_r'
            #ax.imshow(terrain.T, origin='lower', cmap=c_terrain)

        # draw coordinate system
        c_ticks = (242/255,242/255,242/255)
        ratio = local_size_y/5
        tick_length = local_size_y/30
        for i in range(int(local_size_x/ratio)):
            i+=0.5
            x, y = [i*ratio, i*ratio], [0, tick_length]
            ax.plot(x, y, color=c_ticks, linewidth=local_size_y/10)
            ax.text(i*ratio, tick_length*1.2, str(int(i*ratio*yaml_p['unit'])), color=c_ticks, horizontalalignment='center', fontsize=local_size_y)

        for j in range(int(local_size_y/ratio)):
            j+=0.5
            x, y = [0, tick_length], [j*ratio, j*ratio]
            ax.plot(x, y, color=c_ticks, linewidth=local_size_y/10)
            ax.text(tick_length*1.2, j*ratio, str(int(j*ratio*yaml_p['unit'])), color=c_ticks, verticalalignment='center', fontsize=local_size_y)

        # Create a Rectangle patch
        rect = patches.Rectangle((0, 0), local_size_x, local_size_y, linewidth=local_size_y/10, edgecolor=c_ticks, facecolor='none')
        ax.add_patch(rect)

        # save figure
        dpi = 5
        plt.savefig('render/wind_map_' + dim + '.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close()

        # read in image with cv to then crop it
        img = cv2.imread('render/wind_map_' + dim + '.png', cv2.IMREAD_UNCHANGED)

        c = [150, 150, 150, 150]
        indices = np.where(np.all(img >= c, axis=-1))
        border_left = indices[0][0]
        border_right = indices[0][-1]+1
        border_top = indices[1][0]
        border_bottom = indices[1][-1]+1
        img = img[border_left:border_right,border_top:border_bottom]

        cv2.imwrite('render/wind_map_' + dim + '.png', img)
