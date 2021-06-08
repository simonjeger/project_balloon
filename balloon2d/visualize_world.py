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
    size_z = len(tensor[0][0])

    pos_x = int(position[0])
    pos_z = int(position[1])

    pos_x = max(pos_x, 0)
    pos_x = min(pos_x, size_x-1)
    pos_z = max(pos_z, 0)
    pos_z = min(pos_z, size_z-1)

    render_ratio = yaml_p['unit_x']/yaml_p['unit_z']

    for dim in ['xz']:
        if dim == 'xz':
            terrain = tensor[0,:,0]
            dir_x = tensor[-4,:,:]
            dir_y = tensor[-2,:,:]
            sig = tensor[-1,:,:]

        local_size_x = len(dir_x)
        local_size_y = len(dir_x[0])

        fig, ax = plt.subplots(frameon=False, figsize=(local_size_x, local_size_y))
        ax.set_axis_off()

        if dim != 'xy':
            cmap = sns.diverging_palette(145, 300, s=50, center="dark", as_cmap=True)
            ax.imshow(dir_y.T, origin='lower', extent=[0, local_size_x, 0, local_size_y], cmap=cmap, alpha=1, vmin=-5, vmax=5, interpolation='bilinear')

            cmap = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
            ax.imshow(dir_x.T, origin='lower', extent=[0, local_size_x, 0, local_size_y], cmap=cmap, alpha=0.7, vmin=-5, vmax=5, interpolation='bilinear')

        # draw terrain & coordinate system
        c_terrain = (161/255,135/255,93/255) #because plt uses values between 0 and 1
        c_contour = sns.cubehelix_palette(start=1.28, rot=0, dark=0.2, light=0.7, reverse=True, as_cmap=True)
        c_ticks = (242/255,242/255,242/255)

        if dim != 'xy':
            # set parameters
            dpi = 100
            ax.set_aspect(1/render_ratio)

            # plot terrain
            ax.fill_between(np.linspace(0,local_size_x,len(terrain)),terrain, color=c_terrain)

            # create ticks
            ratio = 2*local_size_x/render_ratio
            tick_length = local_size_y/30
            for i in range(int(local_size_x/ratio)):
                i+=0.5
                x, y = [i*ratio, i*ratio], [0, tick_length]
                ax.plot(x, y, color=c_ticks, linewidth=local_size_y/100)
                ax.text(i*ratio, tick_length*1.2, str(int(i*ratio*yaml_p['unit_x'])), color=c_ticks, horizontalalignment='center', fontsize=local_size_y/10)

            ratio = 5*local_size_y/render_ratio
            tick_length /= render_ratio
            for j in range(int(local_size_y/ratio)):
                j+=0.5
                x, y = [0, tick_length], [j*ratio, j*ratio]
                ax.plot(x, y, color=c_ticks, linewidth=local_size_y/100)
                ax.text(tick_length*1.2, j*ratio, str(int(j*ratio*yaml_p['unit_z'])), color=c_ticks, verticalalignment='center', fontsize=local_size_y/10)

            # Create a Rectangle patch
            rect = patches.Rectangle((0, 0), local_size_x, local_size_y, linewidth=local_size_y/100, edgecolor=c_ticks, facecolor='none')
            ax.add_patch(rect)

        else:
            # set parameters
            dpi = 50
            ax.set_aspect(1)

            # plot contour lines
            x = np.arange(0, local_size_x, 1)
            y = np.arange(0, local_size_y, 1)

            x = np.linspace(0, local_size_x+1, local_size_x)
            y = np.linspace(0, local_size_y+1, local_size_y)

            X, Y = np.meshgrid(x, y)
            ax.contourf(X,Y,terrain.T, cmap=c_contour, extend=(0,local_size_x,0,local_size_y))
            ax.quiver(X, Y, dir_x, dir_y, scale=yaml_p['unit_x']/5, headwidth=2.5, width=0.005)

            # create ticks
            ratio = 2*local_size_x/render_ratio
            tick_length = local_size_y/60
            for i in range(int(local_size_x/ratio)):
                i+=0.5
                x, y = [i*ratio, i*ratio], [0, tick_length]
                ax.plot(x, y, color=c_ticks, linewidth=local_size_y/10)
                ax.text(i*ratio, tick_length*1.2, str(int(i*ratio*yaml_p['unit_x'])), color=c_ticks, horizontalalignment='center', fontsize=1.1*local_size_y/1.1)

            for j in range(int(local_size_y/ratio)):
                j+=0.5
                x, y = [0, tick_length], [j*ratio, j*ratio]
                ax.plot(x, y, color=c_ticks, linewidth=local_size_y/10)
                ax.text(tick_length*1.2, j*ratio, str(int(j*ratio*yaml_p['unit_x'])), color=c_ticks, verticalalignment='center', fontsize=1.1*local_size_y/1.1)

            # Create a Rectangle patch
            #rect = patches.Rectangle((0, 0), local_size_x, local_size_y, linewidth=local_size_y/10, edgecolor=c_ticks, facecolor='none')
            #ax.add_patch(rect)

        # save figure
        plt.savefig('render/wind_map_' + dim + '.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close()
