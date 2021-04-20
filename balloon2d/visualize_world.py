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

    for n in range(num):
        terrain = tensor_list[n][0,:,0]
        mean_x = tensor_list[n][-3,:,:]
        mean_z = tensor_list[n][-2,:,:]
        sig_xz = tensor_list[n][-1,:,:]
        size_x = len(mean_x)
        size_z = len(mean_x[0])

        z,x = np.meshgrid(np.arange(0, size_z, 1),np.arange(0, size_x, 1))
        fig, ax = plt.subplots(frameon=False, figsize=(size_x,size_z))
        ax.set_axis_off()
        ax.set_aspect(1)

        # standardise color map
        ceil = 10
        color_quiver = mean_x.copy()
        color_quiver = np.maximum(color_quiver, -ceil)
        color_quiver = np.minimum(color_quiver, ceil)
        color_quiver /= 2*ceil
        color_quiver += 0.5

        cm = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
        #cm = sns.color_palette("icefire", as_cmap=True)
        colors = cm(color_quiver).reshape(size_x*size_z,4)

        # generate quiver
        q = ax.quiver(x, z, mean_x, mean_z, color=colors, scale=50*yaml_p['unit'], headwidth=3, width=0.0015)

        c_terrain = (169/255,163/255,144/255) #because plt uses values between 0 and 1
        c_ticks = (242/255,242/255,242/255)

        # plot terrain
        ax.fill_between(np.linspace(0,size_x,len(terrain)),terrain, color=c_terrain)

        # draw coordinate system
        ratio = size_z/5
        tick_length = size_z/30
        for i in range(int(size_x/ratio)):
            i+=0.5
            x, y = [i*ratio, i*ratio], [0, tick_length]
            ax.plot(x, y, color=c_ticks, linewidth=size_z/10)
            ax.text(i*ratio, tick_length*1.2, str(int(i*ratio*yaml_p['unit'])), color=c_ticks, horizontalalignment='center', fontsize=size_z)

        for j in range(int(size_z/ratio)):
            j+=0.5
            x, y = [0, tick_length], [j*ratio, j*ratio]
            ax.plot(x, y, color=c_ticks, linewidth=size_z/10)
            ax.text(tick_length*1.2, j*ratio, str(int(j*ratio*yaml_p['unit'])), color=c_ticks, verticalalignment='center', fontsize=size_z)

        # Create a Rectangle patch
        rect = patches.Rectangle((0, 0), size_x, size_z, linewidth=size_z/10, edgecolor=c_ticks, facecolor='none')
        ax.add_patch(rect)

        # save figure
        plt.savefig(yaml_p['data_path'] + train_or_test + '/image/wind_map' + str(n).zfill(5) + '.png', dpi = 50, bbox_inches='tight', pad_inches=0)
        plt.close()

        # read in image with cv to then crop it
        img = cv2.imread(yaml_p['data_path'] + train_or_test + '/image/wind_map' + str(n).zfill(5) + '.png', cv2.IMREAD_UNCHANGED)
        border_left = int(np.round(1.74*size_x,0)) #1.79
        border_right = int(np.round(1.73*size_x,0)) #1.73
        border_top = int(np.round(1.7*size_z,0)) #1.89
        border_bottom = int(np.round(1.69*size_z,0)) #1.69
        img = img[border_top:len(img)-border_bottom,border_left:len(img[0])-border_right]

        cv2.imwrite(yaml_p['data_path'] + train_or_test + '/image/wind_map' + str(n).zfill(5) + '.png', img)

        print('visualized ' + str(n+1) + ' of ' + str(num) + ' windmaps')
