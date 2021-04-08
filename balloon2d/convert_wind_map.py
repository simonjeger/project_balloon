from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import netCDF4
from pathlib import Path
import os
import shutil
import imageio
import datetime

# reading the nc file and creating Dataset
nc_terrain = netCDF4.Dataset('data_cosmo/cosmo-1_ethz_ana_const.nc')
nc_wind = netCDF4.Dataset('data_cosmo/cosmo-1_ethz_fcst_2018112300.nc')

print(nc_terrain)

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

            axs[j].set_xlabel('x')
            axs[j].set_ylabel('y')
            axs[j].set_aspect(1)

            raster = [np.min(nc_wind['x_1']), np.max(nc_wind['x_1']), np.min(nc_wind['y_1']), np.max(nc_wind['y_1'])]
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
        raster = [np.min(nc_terrain['x_1']), np.max(nc_terrain['x_1']), np.min(nc_terrain['y_1']), np.max(nc_terrain['y_1'])]
        image = axs[-1].imshow(data, cmap=cmap, extent=raster)

        axs[-1].text(0.95, 0.01, 'HSURF', verticalalignment='bottom', horizontalalignment='right', transform=axs[-1].transAxes, color='white', fontsize=15)
        axs[-1].set_xlabel('x')
        axs[-1].set_ylabel('y')
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
