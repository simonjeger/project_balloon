from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import pandas as pd
import torch
import netCDF4
from pathlib import Path
import os
import shutil
import imageio
import datetime

from utils.extract_cosmo_data import extracter

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

def convert_map():
    Path(yaml_p['process_path'] + 'data_cosmo').mkdir(parents=True, exist_ok=True)
    Path(yaml_p['process_path'] + 'data_cosmo/tensor').mkdir(parents=True, exist_ok=True)
    Path(yaml_p['process_path'] + 'data_cosmo/coord').mkdir(parents=True, exist_ok=True)

    #size_x = 362
    #size_y = 252

    size_x = 200
    size_y = 100
    size_z = 105

    world = np.zeros(shape=(1+4,size_x,size_y,size_z))
    coord = np.zeros(shape=(2,size_x,size_y))

    #min_lat :48.096786
    #max_lat :45.50961
    #min_lon :10.868285
    #max_lon :5.5040283

    step_x = size_x/2*yaml_p['unit_xy']
    step_y = size_y/2*yaml_p['unit_xy']

    center_lat = 46.803198
    center_lon = 8.18615665

    start_lat, start_lon = step(center_lat, center_lon, -step_x, -step_y)
    end_lat, end_lon = step(center_lat, center_lon, step_x, step_y)

    print('------- start at ' + str(np.round(start_lat,3)) + ', '+ str(np.round(start_lon,3)) + ' -------')
    print('------- end at ' + str(np.round(end_lat,3)) + ', '+ str(np.round(end_lon,3)) + ' -------')
    print('------- spanning over ' + str(np.round(np.sqrt((2*step_x)**2 + (2*step_y)**2),1)) + ' m -------')

    step_lat = (end_lat - start_lat)/size_y
    step_lon = (end_lon - start_lon)/size_x

    lowest = 0
    highest = size_z*yaml_p['unit_z']
    step_z = (highest - lowest)/size_z

    ext = extracter(yaml_p['process_path'] + 'data_cosmo/cosmo-1_ethz_fcst_2018112300.nc', terrain_file=yaml_p['process_path'] + 'data_cosmo/cosmo-1_ethz_ana_const.nc')

    t = yaml_p['h']
    for i in range(size_x):
        for j in range(size_y):
            out = ext.extract_cosmo_data(start_lat + j*step_lat, start_lon + i*step_lon, t) #used to be 46.947225, 8.693297, 3
            for k in range(size_z):
                # finding closest quadrant
                q_lat = int(np.argmin(abs(out['lat']-start_lat + i*step_lat))/2)
                q_lon = np.argmin(abs(out['lon'][q_lat]-start_lon + i*step_lon))

                # write down coordinates
                coord[0,i,j] = start_lat + j*step_lat
                coord[1,i,j] = start_lon + i*step_lon

                # write terrain
                world[0,i,j,0] = (out['hsurf'][q_lat,q_lon] - lowest) / (highest - lowest) * size_z

                if step_z*k < out['z'][-1,q_lat,q_lon] - lowest:
                    world[-4,i,j,k] = np.mean(out['wind_x'][0,q_lat,q_lon])
                    world[-3,i,j,k] = np.mean(out['wind_x'][0,q_lat,q_lon])
                    world[-2,i,j,k] = np.mean(out['wind_z'][0,q_lat,q_lon])
                    #world[-1,i,j,k] = np.mean(out['wind_z'][k,q_lat,q_lon]) #add variance later
                else:
                    idx = np.argmin(abs(out['z'][:,q_lat,q_lon] - lowest - step_z*k))
                    world[-4,i,j,k] = np.mean(out['wind_x'][idx,q_lat,q_lon])
                    world[-3,i,j,k] = np.mean(out['wind_y'][idx,q_lat,q_lon])
                    world[-2,i,j,k] = np.mean(out['wind_z'][idx,q_lat,q_lon])
                    #world[-1,i,j,k] = np.mean(out['wind_z'][k,q_lat,q_lon]) #add variance later

        print('converted ' + str(np.round(i/size_x*100,1)) + '% of the wind field into tensor at ' + str(t).zfill(2) + ':00')
    print('------- converted to tensor -------')

    # save
    torch.save(world, yaml_p['process_path'] + 'data_cosmo/tensor/wind_map_CH_' + str(t).zfill(2) + '.pt')
    torch.save(coord, yaml_p['process_path'] + 'data_cosmo/coord/coord_map_CH_' + str(t).zfill(2) + '.pt')

def dist(lat_1, lon_1, lat_2, lon_2):
    R = 6371*1000 #radius of earth in meters
    phi = lat_2 - lat_1
    lam = lon_2 - lon_1
    a = np.sin(phi/2)**2 + np.cos(lat_1) * np.cos(lat_2) * np.sin(lam/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    return d

def step(lat, lon, step_x, step_y):
    R = 6371*1000 #radius of earth in meters
    lat = lat + (step_y/R) * (180/np.pi)
    lon = lon + (step_x/R) * (180/np.pi) / np.cos(lat*np.pi/180)
    return lat, lon

def build_set(num, n_h, train_or_test):
    Path(yaml_p['data_path']).mkdir(parents=True, exist_ok=True)
    Path(yaml_p['data_path'] + train_or_test).mkdir(parents=True, exist_ok=True)
    Path(yaml_p['data_path'] + train_or_test + '/tensor').mkdir(parents=True, exist_ok=True)

    seed_overall = np.random.randint(0,2**32 - 1)
    for h in range(n_h):
        tensor = torch.load(yaml_p['process_path'] + 'data_cosmo/tensor/wind_map_CH_' + str(h).zfill(2) + '.pt')
        coord = torch.load(yaml_p['process_path'] + 'data_cosmo/coord/coord_map_CH_' + str(h).zfill(2) + '.pt')
        size_c = len(tensor)
        size_x = yaml_p['size_x']
        size_y = yaml_p['size_y']
        size_z = yaml_p['size_z']

        global_size_x = len(tensor[0])
        global_size_y = len(tensor[0][0])
        global_size_z = len(tensor[0][0][0])

        seed = seed_overall
        N = int(num/4)
        for o in range(4):
            if o == 0:
                tensor_rot = tensor[:,:,:,:]
            elif o == 1:
                tensor_rot = tensor[:,::-1,:,:]
                tensor_rot[-4,:,:,:] = -tensor_rot[-4,:,:,:]
            elif o == 2:
                tensor_rot = tensor[:,:,::-1,:]
                tensor_rot[-3,:,:,:] = -tensor_rot[-3,:,:,:]
            elif o == 3:
                tensor_rot = tensor[:,::-1,::-1,:]
                tensor_rot[-4:-2,:,:,:] = -tensor_rot[-4:-2,:,:,:]

            for n in range(N):
                world = np.ones(shape=(1+4,size_x,size_y,size_z))*size_z #so that it makes the first "while" for sure
                flat = 3
                center_x = int(len(world[0])/2-flat/2)
                center_y = int(len(world[0][0])/2-flat/2)
                while np.max(world[0][center_x:center_x+flat,center_y:center_y+flat,0]) > (1 - yaml_p['min_space'])*size_z: #only generate maps with enough space
                    np.random.seed(seed)
                    seed += 1
                    idx_x = np.random.randint(0,global_size_x - size_x - 1)
                    np.random.seed(seed)
                    seed += 1
                    idx_y = np.random.randint(0,global_size_y - size_y - 1)

                    world = tensor_rot[:,idx_x:idx_x+size_x, idx_y:idx_y+size_y,:]
                    coord_center = coord[:,idx_x + int(size_x/2), idx_y + int(size_y/2)] #only save the center coordinate (lat,lon)

                digits = 4
                name = str(np.round(coord_center[0],digits)).zfill(digits+2) + '_' + str(np.round(coord_center[1],digits)).zfill(digits+2) + '_' + str(o) + '_' + str(h).zfill(2) + '.pt'
                print(name)
                torch.save(world, yaml_p['data_path'] + train_or_test + '/tensor/' + name)

                print('generated ' + str(o*N + n + 1) + ' of ' + str(num) + ' maps at ' + str(h).zfill(2) + ':00')

def visualize_real_data(dimension):
    # reading the nc file and creating Dataset
    nc_terrain = netCDF4.Dataset(yaml_p['process_path'] + 'data_cosmo/cosmo-1_ethz_ana_const.nc')
    nc_wind = netCDF4.Dataset(yaml_p['process_path'] + 'data_cosmo/cosmo-1_ethz_fcst_2018112300.nc')

    if dimension == 'z':
        N = min(len(nc_wind['U'][0,:,0,:]), len(nc_wind['U'][0,:,:,0]))
    else:
        N = len(nc_wind['U'][:,0,0,0])

    # wind data
    for n in range(N):
        cmap = "cividis"
        fig, axs = plt.subplots(4, figsize=(10, 15), dpi=200)

        images = []
        for j in range(3):

            if j == 0:
                dir = 'U'
            elif j == 1:
                dir = 'V'
            elif j == 2:
                dir = 'W'
            else:
                print('no wind in that direction')

            if dimension == 'z':
                if dir != 'W':
                    axs[j].text(0.95, 0.01, dir + ' at z = ' + str(nc_wind['z_1'][n]), verticalalignment='bottom', horizontalalignment='right', transform=axs[j].transAxes, color='white', fontsize=15)
                else:
                    axs[j].text(0.95, 0.01, dir + ' at z = ' + str(nc_wind['z_3'][n]), verticalalignment='bottom', horizontalalignment='right', transform=axs[j].transAxes, color='white', fontsize=15)
                data = nc_wind[dir][0,n,:,:]
            else:
                time_start = datetime.datetime(2018,11,23,0,0,0)
                axs[j].text(0.95, 0.01, dir + ' at t = ' + str(time_start + datetime.timedelta(0,int(nc_wind['time'][n]))), verticalalignment='bottom', horizontalalignment='right', transform=axs[j].transAxes, color='white', fontsize=15)
                data = nc_wind[dir][n,0,:,:]

            axs[j].set_xlabel('lon')
            axs[j].set_ylabel('lat')
            axs[j].set_aspect(1)

            min_lon = nc_wind['lon_1'][np.min(nc_wind['y_1']), np.min(nc_wind['x_1'])]
            max_lon = nc_wind['lon_1'][np.max(nc_wind['y_1']), np.max(nc_wind['x_1'])]
            min_lat = nc_wind['lat_1'][np.min(nc_wind['y_1']), np.min(nc_wind['x_1'])]
            max_lat = nc_wind['lat_1'][np.max(nc_wind['y_1']), np.max(nc_wind['x_1'])]

            raster = [max_lon, min_lon, max_lat, min_lat] #somehow it's reversed
            images.append(axs[j].imshow(data, cmap=cmap, extent=raster))

        # Find the min and max of all colors for use in setting the color scale.
        if n == 0:
            vmin = min(image.get_array().min() for image in images)
            vmax = max(image.get_array().max() for image in images)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for im in images:
            im.set_norm(norm)

        fig.colorbar(images[0], ax=axs[0:3], orientation='vertical', fraction=.1)

        #terrain data
        data = nc_terrain['HSURF'][:,:]

        min_lon = nc_terrain['lon_1'][np.min(nc_terrain['y_1']), np.min(nc_terrain['x_1'])]
        max_lon = nc_terrain['lon_1'][np.max(nc_terrain['y_1']), np.max(nc_terrain['x_1'])]
        min_lat = nc_terrain['lat_1'][np.min(nc_terrain['y_1']), np.min(nc_terrain['x_1'])]
        max_lat = nc_terrain['lat_1'][np.max(nc_terrain['y_1']), np.max(nc_terrain['x_1'])]

        raster = [max_lon, min_lon, max_lat, min_lat] #somehow it's reversed
        image = axs[-1].imshow(data, cmap=cmap, extent=raster)

        axs[-1].text(0.95, 0.01, 'HSURF', verticalalignment='bottom', horizontalalignment='right', transform=axs[-1].transAxes, color='white', fontsize=15)
        axs[-1].set_xlabel('lon')
        axs[-1].set_ylabel('lat')
        axs[-1].set_aspect(1)
        fig.colorbar(image, ax=axs[3], orientation='vertical', fraction=.1)

        # Make images respond to changes in the norm of other images (e.g. via the
        # "edit axis, curves and images parameters" GUI on Qt), but be careful not to
        # recurse infinitely!
        def update(changed_image):
            for im in images:
                if (changed_image.get_cmap() != im.get_cmap()
                        or changed_image.get_clim() != im.get_clim()):
                    im.set_cmap(changed_image.get_cmap())
                    im.set_clim(changed_image.get_clim())

        for im in images:
            im.callbacksSM.connect('changed', update)

        # Build folder structure if it doesn't exist yet
        path = yaml_p['process_path'] + 'data_cosmo/temp'
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(path + '/gif_' + str(n).zfill(5) + '.png')
        plt.close()
        print('saving frame nr. ' + str(n))

    # Build GIF
    with imageio.get_writer(yaml_p['process_path'] + 'data_cosmo/cosmo_' + dimension + '.gif', mode='I') as writer:
        name_list = os.listdir(path)
        name_list.sort()
        for name in name_list:
            print('writing ' + name + ' into gif')
            image = imageio.imread(path + '/' + name)
            writer.append_data(image)

    # Delete temp folder
    shutil.rmtree(path)

#visualize_real_data('z')
#visualize_real_data('time')
convert_map()

#build_set(10, 7, 'train')
#build_set(10, 7, 'test')
