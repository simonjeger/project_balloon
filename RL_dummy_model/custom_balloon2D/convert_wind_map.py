from matplotlib import colors
import matplotlib.pyplot as plt
matplotlib.use("Agg") # is needed for processing on cluster
import numpy as np
import pandas as pd
import netCDF4
from pathlib import Path
import os
import shutil
import imageio

#fp='data/real/cosmo-1_ethz_ana_const.nc' # your file name with the eventual path
fp='data/real/cosmo-1_ethz_fcst_2018112300.nc'
nc = netCDF4.Dataset(fp) # reading the nc file and creating Dataset

for n in range(min(len(nc['U'][1,:,0,:]), len(nc['U'][1,:,:,0]))):
    Nr = 2
    Nc = 3
    cmap = "cividis"

    fig, axs = plt.subplots(Nr, Nc, figsize=(10, 15), dpi=100)
    fig.suptitle('Wind data with orgin at [' + str(n) + ',' + str(n) + ']')

    images = []
    for j in range(Nc):
        if j == 0:
            dir = 'U'
        elif j == 1:
            dir = 'V'
        elif j == 2:
            dir = 'W'
        else:
            print('no wind in that direction')
        for i in range(Nr):
            # Generate data with a range that varies from one plot to the next.
            if i == 0:
                data = nc[dir][n,:,0,:].transpose()
                axs[i,j].set_title(dir)
                axs[i,j].set_xlabel('x')
                axs[i,j].set_ylabel('z')
                if j != 'W':
                    raster = [min(nc['x_1']), max(nc['x_1']), min(nc['z_1']), max(nc['z_1'])]
                else:
                    raster = [min(nc['x_1']), max(nc['x_1']), min(nc['z_3']), max(nc['z_3'])]
                images.append(axs[i, j].imshow(data, cmap=cmap, extent=raster))
            elif i == 1:
                data = nc[dir][n,:,:,0].transpose()
                axs[i,j].set_title(dir)
                axs[i,j].set_xlabel('y')
                axs[i,j].set_ylabel('z')
                if j != 'W':
                    raster = [min(nc['y_1']), max(nc['y_1']), min(nc['z_1']), max(nc['z_1'])]
                else:
                    raster = [min(nc['y_1']), max(nc['y_1']), min(nc['z_3']), max(nc['z_3'])]
                images.append(axs[i, j].imshow(data, cmap=cmap, extent=raster))
            else:
                print('no wind in that dimension')
            #axs[i, j].label_outer()

    # Find the min and max of all colors for use in setting the color scale.
    if n == 0:
        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)

    fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)


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
    path = 'data/real/temp'
    Path(path).mkdir(parents=True, exist_ok=True)
    plt.savefig(path + '/gif_' + str(n).zfill(3) + '.png')
    plt.close()
    print('saving frame nr. ' + str(n))

# Build GIF
with imageio.get_writer('data/real/mygif.gif', mode='I') as writer:
    name_list = os.listdir(path)
    print(name_list)
    print(name_list.sort())
    for name in name_list:
        print('writing ' + name + ' into gif')
        image = imageio.imread(path + '/' + name)
        writer.append_data(image)

# Delete temp folder
shutil.rmtree(path)
