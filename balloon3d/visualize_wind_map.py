import numpy as np
import torch
from random import gauss
import matplotlib.pyplot as plt
import os
import cv2

def visualize_wind_map(tensor, position):

    # initializing basic operations
    u = tensor[:,:,:,0]
    v = tensor[:,:,:,1]
    w = tensor[:,:,:,2]
    pos_x = int(position[0])
    pos_y = int(position[1])
    pos_z = int(position[2])
    turb = tensor[:,:,:,3]
    size_x = len(u)
    size_y = len(u[0])
    size_z = len(u[0][0])

    if (pos_x >= 0) & (pos_y >= 0) & (pos_z >= 0) & (pos_x < size_x) & (pos_y < size_y) & (pos_z < size_z):

        x_ = np.arange(0, size_x, 1)
        y_ = np.arange(0, size_y, 1)
        z_ = np.arange(0, size_z, 1)
        y, x, z = np.meshgrid(x_, y_, z_) # because somehow meshgrid works like that (trial and error)

        # initializing three times the same figure (for each case)
        fig, ax_xz = plt.subplots(frameon=False)
        ax_xz.set_axis_off()
        ax_xz.set_aspect(1)
        ax_xz.quiver(x[:,pos_y,:],z[:,pos_y,:],u[:,pos_y,:],w[:,pos_y,:],turb[:,pos_y,:])
        plt.savefig('render/wind_map_xz.png', dpi = 100, bbox_inches='tight', pad_inches=0)
        plt.close()

        fig, ax_yz = plt.subplots(frameon=False)
        ax_yz.set_axis_off()
        ax_yz.set_aspect(1)
        ax_yz.quiver(y[pos_x,:,:],z[pos_x,:,:],v[pos_x,:,:],w[pos_x,:,:],turb[pos_x,:,:])
        plt.savefig('render/wind_map_yz.png', dpi = 100, bbox_inches='tight', pad_inches=0)
        plt.close()

        fig, ax_xy = plt.subplots(frameon=False)
        ax_xy.set_axis_off()
        ax_xy.set_aspect(1)
        ax_xy.quiver(x[:,:,pos_z],y[:,:,pos_z],u[:,:,pos_z],v[:,:,pos_z],turb[:,:,pos_z])
        plt.savefig('render/wind_map_xy.png', dpi = 100, bbox_inches='tight', pad_inches=0)
        plt.close()

"""
    # read in image with cv to then crop it
    img_xz = cv2.imread('data/' + train_or_test + '/image/wind_map_xz' + str(n).zfill(5) + '.png', cv2.IMREAD_UNCHANGED)
    img_yz = cv2.imread('data/' + train_or_test + '/image/wind_map_yz' + str(n).zfill(5) + '.png', cv2.IMREAD_UNCHANGED)
    img_xy = cv2.imread('data/' + train_or_test + '/image/wind_map_xy' + str(n).zfill(5) + '.png', cv2.IMREAD_UNCHANGED)

    border_x = int(0.9*size_x)
    border_y = int(0.4*size_z)

    img_xz = img_xz[border_y:len(img)-border_y,border_x:len(img[0])-border_x]
    img_yz = img_yz[border_y:len(img)-border_y,border_x:len(img[0])-border_x]
    img_xy = img_xy[border_y:len(img)-border_y,border_x:len(img[0])-border_x]

    cv2.imwrite('data/' + train_or_test + '/image/wind_map_xz' + str(n).zfill(5) + '.png', img)
    cv2.imwrite('data/' + train_or_test + '/image/wind_map_yz' + str(n).zfill(5) + '.png', img)
    cv2.imwrite('data/' + train_or_test + '/image/wind_map_xy' + str(n).zfill(5) + '.png', img)
"""
