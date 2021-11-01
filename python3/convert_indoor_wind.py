import torch
import numpy as np
from random import gauss
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cv2
import random
from pathlib import Path

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

def single_prop():
    u = np.zeros((11,9))
    u[:,0] = [1.7, 1.8, 1.6, 1.2, 0.9, 0.7, 0.5, 0.4, 0.4, 0.0, 0.0]
    u[:,1] = [2.1, 1.7, 1.5, 1.3, 1.1, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3]
    u[:,2] = [1.7, 1.8, 1.3, 1.0, 0.9, 0.8, 0.7, 0.7, 0.6, 0.5, 0.4]
    u[:,3] = [0.6, 1.0, 0.9, 0.9, 0.7, 0.7, 0.6, 0.6, 0.6, 0.5, 0.4]
    u[:,4] = [0.0, 0.6, 0.7, 0.6, 0.6, 0.6, 0.6, 0.6, 0.5, 0.4, 0.4]
    u[:,5] = [0.0, 0.0, 0.0, 0.4, 0.5, 0.6, 0.5, 0.6, 0.5, 0.0, 0.0]
    u[:,6] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    u[:,7] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    u[:,8] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    grid_vec = []
    u_vec = []
    for i in range(11):
        for j in range(8):
            grid_vec.append([i,j])
            u_vec.append(u[i,j])

    interp_radial = LinearNDInterpolator(grid_vec, u_vec, fill_value=0)

    unit_grid_x = 0.5
    unit_grid_y = 0.1
    unit_grid_z = 0.1

    grid_vec = []
    u_vec = []
    for i in range(11):
        for j in range(17):
            for k in range(17):
                grid_vec.append([i*unit_grid_x,(j-8)*unit_grid_y,(k-8)*unit_grid_z])
                u_vec.append(interp_radial([i,np.sqrt((j-8)**2+(k-8)**2)]))

    return u_vec, grid_vec

def approx_wind_machine():
    u_vec, grid_vec = single_prop()
    interp = LinearNDInterpolator(grid_vec, u_vec, fill_value=0)

    unit_grid_x = 0.5
    unit_grid_y = 0.1
    unit_grid_z = 0.1

    u_vec_new = []
    u_grid_new = []
    for i in range(11):
        for j in range(36):
            for k in range(17):
                u_grid_new.append([i*unit_grid_x,(j-18)*unit_grid_y,(k-8)*unit_grid_z])
                u_vec_new.append(interp([i*unit_grid_x,(j-8)*unit_grid_y,(k-8)*unit_grid_z]) + interp([i*unit_grid_x,(j-8-1*20/3)*unit_grid_y,(k-8)*unit_grid_z]) + interp([i*unit_grid_x,(j-8-2*20/3)*unit_grid_y,(k-8)*unit_grid_z]) + interp([i*unit_grid_x,(j-8-3*20/3)*unit_grid_y,(k-8)*unit_grid_z]))

    return u_vec_new, u_grid_new

def generate_world(num, n_t, train_or_test):
    Path(yaml_p['data_path']).mkdir(parents=True, exist_ok=True)
    Path(yaml_p['data_path'] + train_or_test).mkdir(parents=True, exist_ok=True)
    Path(yaml_p['data_path'] + train_or_test + '/tensor').mkdir(parents=True, exist_ok=True)

    size_x = yaml_p['size_x']
    size_y = yaml_p['size_y']
    size_z = yaml_p['size_z']

    #interpolation
    model,grid = approx_wind_machine()
    interp = LinearNDInterpolator(grid, model, fill_value=0)

    for n in range(num):
        world = np.zeros(shape=(1+4,size_x,size_y,size_z))
        center_x = int(len(world[0])/2)
        center_y = int(len(world[0][0])/2)

        terrain = generate_terrain(size_x, size_y, size_z)

        while np.max(world[1:3,center_x,center_y,:]) < 0.4: #only generate maps with enough variaty above the orgin
            wind = generate_wind(size_x, size_y, size_z, terrain, model, grid, interp)

            for i in range(size_x):
                for j in range(size_y):
                    world[0,i,j,0] = terrain[i,j]

                    for k in range(size_z):
                        if k >= np.floor(terrain[i,j]):
                            world[-4,i,j,k] = wind[0][i,j,k]
                            world[-3,i,j,k] = wind[1][i,j,k]
                            world[-2,i,j,k] = wind[2][i,j,k]
                            world[-1,i,j,k] = wind[3][i,j,k]

        # save
        for t in range(n_t):
            torch.save(world, yaml_p['data_path'] + train_or_test + '/tensor/wind_map' + str(n).zfill(5) + '_' + str(t).zfill(2) + '.pt')
        print('generated ' + str(n+1) + ' of ' + str(num) + ' worlds')

def generate_terrain(size_x, size_y, size_z):
    # generate mean & uncertainty
    terrain = np.zeros(shape=(size_x,size_y))

    min_sky = size_z/10*2
    m = int(size_x)
    magnitude = 100
    for i in range(m):
        pos_x = random.randint(0, size_x-1)
        pos_y = random.randint(0, size_y-1)
        terrain[pos_x, pos_y] = gauss(magnitude,size_z/5) #0,2

    terrain = gaussian_filter(terrain, sigma = 10)

    for i in range(len(terrain)):
        for j in range(len(terrain[0])):
            terrain[i,j] = min(terrain[i,j], size_z - min_sky)
            terrain[i,j] = max(terrain[i,j], 0)

    # for homogeneous field
    terrain *= 0
    return terrain

def generate_wind(size_x, size_y, size_z, terrain, model, grid, interp):
    unit_wind_x = yaml_p['unit_xy']
    unit_wind_y = yaml_p['unit_xy']
    unit_wind_z = yaml_p['unit_z']

    wind = np.zeros((4, size_x, size_y, size_z))

    N = 2
    for n in range(N):
        side = np.random.randint(0,3)
        location = np.random.uniform(0,1)
        height = np.random.uniform(0,1)*size_z
        angle = np.random.uniform(-1,1)*np.pi/4
        scale = np.random.uniform(0.25,0.5)

        if side == 0:
            o_x = 0*size_x - abs(np.sin(angle))*18*0.1/unit_wind_y
            o_y = location*size_y
            o_angle = 0

        elif side == 1:
            o_x = location*size_x
            o_y = 0*size_y - abs(np.sin(angle))*18*0.1/unit_wind_x
            o_angle = np.pi/2

        elif side == 2:
            o_x = 1*size_x + abs(np.sin(angle))*18*0.1/unit_wind_y
            o_y = location*size_y
            o_angle = np.pi

        elif side == 3:
            o_x = location*size_x
            o_y = 1*size_y + abs(np.sin(angle))*18*0.1/unit_wind_x
            o_angle = np.pi*3/2

        a = np.cos(angle+o_angle)
        b = -np.sin(angle+o_angle)
        c = np.sin(angle+o_angle)
        d = np.cos(angle+o_angle)

        for i in range(size_x):
            for j in range(size_y):
                for k in range(size_z):
                    c_x = 1/a*(i+b/d*(o_y - j) - o_x)/(1 - (b*c)/(a*d))
                    c_y = 1/d*(j+c/a*(o_x - i) - o_y)/(1 - (b*c)/(a*d))
                    c_z = k - height
                    mag = interp([c_x*unit_wind_x, c_y*unit_wind_y, c_z*unit_wind_z])
                    wind[0,i,j,k] += a*mag*scale
                    wind[1,i,j,k] += c*mag*scale

    wind = gaussian_filter(wind, sigma=0.1)
    return wind

generate_world(500, 7, 'train')
generate_world(500, 7, 'test')
