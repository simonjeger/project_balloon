import datetime as dt
import meteomatics.api as api
import numpy as np
import time
import torch
import datetime
from pathlib import Path

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

def step(lat, lon, step_x, step_y):
    R = 6371*1000 #radius of earth in meters
    lat = lat + (step_y/R) * (180/np.pi)
    lon = lon + (step_x/R) * (180/np.pi) / np.cos(lat*np.pi/180)
    return lat, lon

center_lat = 47.008705
center_lon = 7.174884

print("Write 0 for position to latlon, 1 for latlon to position")
mode = input()
if float(mode) == 0:
    while True:
        print('Write the x-position')
        pos_x = float(input())
        print('Write the y-position')
        pos_y = float(input())

        step_x = (pos_x-(yaml_p['size_x']-1)/2)*yaml_p['unit_xy']
        step_y = (pos_y-(yaml_p['size_y']-1)/2)*yaml_p['unit_xy']

        res = step(center_lat, center_lon, step_x, step_y)
        print('Latitude, longitude: ' + str(res[0]) + ' ' + str(res[1]))

elif float(mode) == 1:
    print(ERROR this is not done yet)
    print('Write a latitude')
    lat = input()
    print('Write a longitude')
else:
    print('Please type either 0 or 1')
