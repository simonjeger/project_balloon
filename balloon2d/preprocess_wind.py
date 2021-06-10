import numpy as np

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

def squish(data, ceiling):
    size_x = len(data[0])
    size_z = len(data[0][0])
    res_z = size_z

    data_squished = np.zeros((len(data),size_x,res_z))
    for i in range(size_x):
        bottom = data[0,i,0]
        top = ceiling

        x_old = np.arange(0,size_z,1)
        x_new = np.linspace(bottom,top,res_z)
        data_squished[0,:,0] = data[0,:,0] #terrain stays the same

        for j in range(1,len(data)):
            data_squished[j,i,:] = np.interp(x_new,x_old,data[j,i,:])

    return data_squished

def unsquish(data_squished, ceiling):
    size_x = len(data_squished[0])
    size_z = len(data_squished[0][0])

    data = np.zeros((len(data_squished),size_x,size_z))
    for i in range(size_x):
        bottom = data_squished[0,i,0]
        top = ceiling

        x_old = np.linspace(bottom,top,len(data_squished[0,0,:]))
        x_new = np.arange(0,size_z,1)
        data[0,:,0] = data_squished[0,:,0] #terrain stays the same

        for j in range(1,len(data_squished)):
            data[j,i,:] = np.interp(x_new,x_old,data_squished[j,i,:])

    return data
