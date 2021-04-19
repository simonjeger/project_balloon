from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import netCDF4
from pathlib import Path
import os
import shutil
import imageio
import datetime

from extract_cosmo_data import extract_cosmo_data

size_x = 3000
size_z = 1000

world = np.zeros(shape=(1+3,size_x,size_z))

start_lat = 47.3769
start_lon = 8.5417
end_lat = 46.9480
end_lon = 7.4474

step_lat = (end_lat - start_lat)/size_x
step_lon = (end_lon - start_lon)/size_x

# finding lowest point in terrain
lowest = np.inf
highest = -np.inf
"""
for i in range(size_x):
    out = extract_cosmo_data('data_cosmo/cosmo-1_ethz_fcst_2018112300.nc', start_lat + i*step_lat, start_lon + i*step_lon, 3, terrain_file='data_cosmo/cosmo-1_ethz_ana_const.nc') #used to be 46.947225, 8.693297, 3
    q_lat = int(np.argmin(abs(out['lat']-start_lat + i*step_lat))/2)
    q_lon = np.argmin(abs(out['lon'][q_lat]-start_lon + i*step_lon))

    if out['hsurf'][q_lat,q_lon] < lowest:
        lowest = out['hsurf'][q_lat,q_lon]
    if out['hsurf'][q_lat,q_lon] > highest:
        highest = out['hsurf'][q_lat,q_lon]

    print('looked in ' + str(np.round(i/size_x*100,1)) + '% of the terrain for the lowest and highest point')
print('------- lowest point at ' + str(lowest) + ' m, highest point at ' + str(highest) + ' m -------')

#lowest = 392.345703125
#highest = 780.470703125

step_z = (3300 + highest - lowest)/size_z

for i in range(size_x):
    out = extract_cosmo_data('data_cosmo/cosmo-1_ethz_fcst_2018112300.nc', start_lat + i*step_lat, start_lon + i*step_lon, 3, terrain_file='data_cosmo/cosmo-1_ethz_ana_const.nc') #used to be 46.947225, 8.693297, 3
    for j in range(size_z):

        # finding closest quadrant
        q_lat = int(np.argmin(abs(out['lat']-start_lat + i*step_lat))/2)
        q_lon = np.argmin(abs(out['lon'][q_lat]-start_lon + i*step_lon))

        # write terrain
        world[0,i,0] = (out['hsurf'][q_lat,q_lon] - lowest) / (3300 + highest - lowest) * size_z
        if step_z*j >= out['z'][-1,q_lat,q_lon] - lowest:
            idx = np.argmin(abs(out['z'][:,q_lat,q_lon] - lowest - step_z*j))
            world[-3,i,j] = np.mean(out['wind_x'][idx,q_lat,q_lon])
            world[-2,i,j] = np.mean(out['wind_z'][idx,q_lat,q_lon])
            #world[-1,i,j] = np.mean(out['wind_z'][j,q_lat,q_lon])

    print('converted ' + str(np.round(i/size_x*100,1)) + '% of the wind field into tensor')
print('------- converted to tensor -------')

# save
#torch.save(world, 'data_cosmo/tensor/wind_map' + str(n).zfill(5) + '.pt')
torch.save(world, 'data_cosmo/tensor/wind_map_0.pt')
"""


# reading the nc file and creating Dataset
nc_terrain = netCDF4.Dataset('data_cosmo/cosmo-1_ethz_ana_const.nc')
nc_wind = netCDF4.Dataset('data_cosmo/cosmo-1_ethz_fcst_2018112300.nc')

def visualize_real_data(dimension):
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

visualize_real_data('z')
visualize_real_data('time')
