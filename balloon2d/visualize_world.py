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

def visualize_world(train_or_test):
    name_list = os.listdir(yaml_p['data_path'] + train_or_test + '/tensor/')
    name_list.sort()
    tensor_list = []
    for name in name_list:
        tensor_list.append(torch.load(yaml_p['data_path'] + train_or_test + '/tensor/' + name))

    num = len(tensor_list)

    render_ratio = int(yaml_p['unit_xy'] / yaml_p['unit_z'])

    for n in range(num):
        terrain = tensor_list[n][0,:,0]
        mean_x = tensor_list[n][-3,:,:]
        mean_z = tensor_list[n][-2,:,:]
        sig_xz = tensor_list[n][-1,:,:]
        size_x = len(mean_x)
        size_z = len(mean_x[0])

        z,x = np.meshgrid(np.arange(0, size_z, 1),np.arange(0, size_x, 1))
        fig, ax = plt.subplots(frameon=False, figsize=(size_x,size_z))


        #cmap = sns.diverging_palette(220, 20, as_cmap=True)
        cmap = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
        ax.imshow(mean_x.T, origin='lower', extent=[0, size_x, 0, size_z], cmap=cmap, alpha=0.5, vmin=-5, vmax=5, interpolation='bilinear')

        #cmap = sns.diverging_palette(145, 300, s=60, as_cmap=True)
        cmap = sns.diverging_palette(145, 300, s=50, center="dark", as_cmap=True)
        ax.imshow(mean_z.T, origin='lower', extent=[0, size_x, 0, size_z], cmap=cmap, alpha=0.5, vmin=-5, vmax=5, interpolation='bilinear')

        ax.set_axis_off()
        ax.set_aspect(1/render_ratio)

        c_terrain = (169/255,163/255,144/255) #because plt uses values between 0 and 1
        c_ticks = (242/255,242/255,242/255)

        # plot terrain
        ax.fill_between(np.linspace(0,size_x,len(terrain)),terrain, color=c_terrain)

        # draw coordinate system
        ratio = 2*size_x/render_ratio
        tick_length = size_z/30
        for i in range(int(size_x/ratio)):
            i+=0.5
            x, y = [i*ratio, i*ratio], [0, tick_length]
            ax.plot(x, y, color=c_ticks, linewidth=size_z/100)
            ax.text(i*ratio, tick_length*1.2, str(int(i*ratio*yaml_p['unit_xy'])), color=c_ticks, horizontalalignment='center', fontsize=size_z/20)

        ratio = 5*size_z/render_ratio
        tick_length /= render_ratio
        for j in range(int(size_z/ratio)):
            j+=0.5
            x, y = [0, tick_length], [j*ratio, j*ratio]
            ax.plot(x, y, color=c_ticks, linewidth=size_z/100)
            ax.text(tick_length*1.2, j*ratio, str(int(j*ratio*yaml_p['unit_z'])), color=c_ticks, verticalalignment='center', fontsize=size_z/20)

        # Create a Rectangle patch
        rect = patches.Rectangle((0, 0), size_x, size_z, linewidth=size_z/100, edgecolor=c_ticks, facecolor='none')
        ax.add_patch(rect)

        # save figure
        plt.savefig(yaml_p['data_path'] + train_or_test + '/image/wind_map' + str(n).zfill(5) + '.png', dpi = 150, bbox_inches='tight', pad_inches=0)
        plt.close()

        print('visualized ' + str(n+1) + ' of ' + str(num) + ' windmaps')
