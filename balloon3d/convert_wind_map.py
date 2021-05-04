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

from utils.extract_cosmo_data import extract_cosmo_data

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

def convert_map():
    size_x = 362
    size_y = 261
    size_z = 105

    world = np.zeros(shape=(1+3,size_x,size_y,size_z))

    #min_lat :48.096786
    #max_lat :45.50961
    #min_lon :10.868285
    #max_lon :5.5040283

    center_lat = 46.803198
    center_lon = 8.18615665
    step_x = size_x/2*yaml_p['unit_xy']
    step_y = size_y/2*yaml_p['unit_xy']

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

    for i in range(size_x):
        for j in range(size_y):
            out = extract_cosmo_data('data_cosmo/cosmo-1_ethz_fcst_2018112300.nc', start_lat + j*step_lat, start_lon + i*step_lon, 3, terrain_file='data_cosmo/cosmo-1_ethz_ana_const.nc') #used to be 46.947225, 8.693297, 3
            for k in range(size_z):
                # finding closest quadrant
                q_lat = int(np.argmin(abs(out['lat']-start_lat + i*step_lat))/2)
                q_lon = np.argmin(abs(out['lon'][q_lat]-start_lon + i*step_lon))

                # write terrain
                world[0,i,j,0] = (out['hsurf'][q_lat,q_lon] - lowest) / (3300 + highest - lowest) * size_z

                if step_z*k >= out['z'][-1,q_lat,q_lon] - lowest:
                    idx = np.argmin(abs(out['z'][:,q_lat,q_lon] - lowest - step_z*k))
                    world[-4,i,j,k] = np.mean(out['wind_x'][idx,q_lat,q_lon])
                    world[-3,i,j,k] = np.mean(out['wind_y'][idx,q_lat,q_lon])
                    world[-2,i,j,k] = np.mean(out['wind_z'][idx,q_lat,q_lon])
                    #world[-1,i,j,k] = np.mean(out['wind_z'][k,q_lat,q_lon]) #add variance later

                    """
                    # interpolation in z
                    out_lat = out['lat'].reshape(4)
                    out_lon = out['lon'].reshape(4)
                    out_z = out['z'].reshape(-1,4)
                    out_wind_x = out['wind_x'].reshape(-1,4)
                    out_wind_y = out['wind_y'].reshape(-1,4)
                    out_wind_z = out['wind_z'].reshape(-1,4)

                    # interpolate over z
                    interp_z = np.zeros(4)
                    for p in range(4):
                        interp_z[p] = np.interp(lowest + step_z*k, out_z[:,p], out_wind_x[:,p])

                    from scipy.interpolate import interpn
                    from scipy.interpolate import griddata

                    points = (out_lat, out_lon)
                    values = interp_z
                    values = np.meshgrid(*values)
                    point = np.array([start_lat + j*step_lat, start_lon + i*step_lon])
                    grid_x, grid_y = np.mgrid[min(out_lat):max(out_lat):100j, min(out_lon):max(out_lon):100j]
                    grid = griddata(points, values, (grid_x, grid_y), method='linear')
                    print(points)
                    print(values)
                    print(grid)

                    print('----------------------------------------------------')


                    x = [out['z'], out['lat'], out['lon']]
                    y = np.zeros((len(out['wind_x']),2,2))
                    for l in range(len(out['wind_x'])):
                        for m in range(2):
                            for n in range(2):
                                y[l,m,n] = out['wind_x'][l]

                    world[-4,i,j,k] = scipy.interpolate.interpn((step_z*k, start_lat + j*step_lat, start_lon + i*step_lon), x, out['wind_x'])
                    """
        torch.save(world, 'data_cosmo/tensor/wind_map_intsave_0.pt')
        print('converted ' + str(np.round(i/size_x*100,1)) + '% of the wind field into tensor')
    print('------- converted to tensor -------')

    # save
    torch.save(world, 'data_cosmo/tensor/wind_map_CH.pt')

def dist(lat_1, lon_1, lat_2, lon_2):
    R = 6371*1000 #radius of earth in meters
    phi = lat_2 - lat_1
    lam = lon_2 - lon_1
    a = np.sin(phi/2)**2 + np.cos(lat_1) * np.cos(lat_2) * sin(lam/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    return d

def step(lat, lon, step_x, step_y):
    R = 6371*1000 #radius of earth in meters
    lat = lat + (step_y/R) * (180/np.pi)
    lon = lon + (step_x/R) * (180/np.pi) / np.cos(lat*np.pi/180)
    return lat, lon

def build_set(num):
    tensor = torch.load('data_cosmo/tensor/wind_map_CH.pt')
    size_c = len(tensor)
    size_x = yaml_p['size_x']
    size_y = yaml_p['size_y']
    size_z = yaml_p['size_z']

    global_size_x = len(tensor[0])
    global_size_y = len(tensor[0][0])
    global_size_z = len(tensor[0][0][0])

    for n in range(num):
        idx_x = np.random.randint(0,global_size_x - size_x - 1)
        idx_y = np.random.randint(0,global_size_y - size_y - 1)

        world = tensor[:,idx_x:idx_x+size_x, idx_y:idx_y+size_y,:]

        torch.save(world, yaml_p['data_path'] + train_or_test + '/tensor/wind_map' + str(n).zfill(5) + '.pt')
        print('generated ' + str(n+1) + ' of ' + str(num) + ' sets')

def visualize_real_data(dimension):
    # reading the nc file and creating Dataset
    nc_terrain = netCDF4.Dataset('data_cosmo/cosmo-1_ethz_ana_const.nc')
    nc_wind = netCDF4.Dataset('data_cosmo/cosmo-1_ethz_fcst_2018112300.nc')

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
        path = 'data_cosmo/temp'
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(path + '/gif_' + str(n).zfill(5) + '.png')
        plt.close()
        print('saving frame nr. ' + str(n))

    # Build GIF
    with imageio.get_writer('data_cosmo/cosmo_' + dimension + '.gif', mode='I') as writer:
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
#convert_map()
#build_set(10, 'train')
