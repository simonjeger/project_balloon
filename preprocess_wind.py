import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)


def preprocess(data,ceiling):
    # interpolate wind in squished frame
    data_squished = squish(data,ceiling)

    squished_size_x = len(data_squished[0])
    squished_size_y = len(data_squished[0][0])
    squished_size_z = len(data_squished[0][0][0])

    x = np.arange(squished_size_x)
    y = np.arange(squished_size_y)
    z = np.arange(squished_size_z)
    f_squished_0 = RegularGridInterpolator((x, y, z), data_squished[0])
    f_squished_1 = RegularGridInterpolator((x, y, z), data_squished[1])
    f_squished_2 = RegularGridInterpolator((x, y, z), data_squished[2])
    f_squished_3 = RegularGridInterpolator((x, y, z), data_squished[3])
    f_squished_4 = RegularGridInterpolator((x, y, z), data_squished[4])


    upsample_factor = yaml_p['unit_xy']/yaml_p['unit_z']
    res = 1
    x_p = np.linspace(0, squished_size_x-1, int(squished_size_x*upsample_factor*res))
    y_p = np.linspace(0, squished_size_y-1, int(squished_size_y*upsample_factor*res))
    z_p = np.linspace(0, squished_size_z-1, int(squished_size_z*res))

    points = np.vstack(np.meshgrid(x_p,y_p,z_p)).reshape(3,-1).T
    data_squished_0 = f_squished_0(points)*res
    data_squished_1 = f_squished_1(points)
    data_squished_2 = f_squished_2(points)
    data_squished_3 = f_squished_3(points)
    data_squished_4 = f_squished_4(points)

    data_squished_0 = data_squished_0.reshape(int(squished_size_x*upsample_factor*res),int(squished_size_y*upsample_factor*res),int(squished_size_z*res))
    data_squished_1 = data_squished_1.reshape(int(squished_size_x*upsample_factor*res),int(squished_size_y*upsample_factor*res),int(squished_size_z*res))
    data_squished_2 = data_squished_2.reshape(int(squished_size_x*upsample_factor*res),int(squished_size_y*upsample_factor*res),int(squished_size_z*res))
    data_squished_3 = data_squished_3.reshape(int(squished_size_x*upsample_factor*res),int(squished_size_y*upsample_factor*res),int(squished_size_z*res))
    data_squished_4 = data_squished_4.reshape(int(squished_size_x*upsample_factor*res),int(squished_size_y*upsample_factor*res),int(squished_size_z*res))

    data_squished = np.array([data_squished_0, data_squished_1, data_squished_2, data_squished_3, data_squished_4])
    data = unsquish(data_squished,ceiling*res)

    torch.save(data, 'render/world_squished.pt')
    return data

def squish(data, ceiling):
    size_x = len(data[0])
    size_y = len(data[0][0])
    size_z = len(data[0][0][0])
    res_z = size_z

    data_squished = np.zeros((len(data),size_x,size_y,res_z))
    for i in range(size_x):
        for j in range(size_y):
            bottom = data[0,i,j,0]
            top = ceiling

            x_old = np.arange(0,size_z,1)
            x_new = np.linspace(bottom,top,res_z)
            data_squished[0,:,:,0] = data[0,:,:,0] #terrain stays the same

            for k in range(1,len(data)):
                data_squished[k,i,j,:] = np.interp(x_new,x_old,data[k,i,j,:])
    return data_squished

def unsquish(data_squished, ceiling):
    size_x = len(data_squished[0,:,0,0])
    size_y = len(data_squished[0,0,:,0])
    size_z = len(data_squished[0,0,0,:])
    data = np.zeros((len(data_squished),size_x,size_y,size_z))
    for i in range(size_x):
        print(np.round(100*i/size_x,0))
        for j in range(size_y):
            bottom = data_squished[0,i,j,0]
            top = ceiling

            x_old = np.linspace(bottom,top,size_z)
            x_new = np.arange(0,size_z,1)
            data[0,:,:,0] = data_squished[0,:,:,0] #terrain stays the same

            for k in range(1,len(data_squished)):
                data[k,i,j,:] = np.interp(x_new,x_old,data_squished[k,i,j,:])

    return data
