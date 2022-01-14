import datetime as dt
import meteomatics.api as api
import numpy as np
import time
import torch

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

# Credentials:
username = 'ethz_jeger'
password = 'yu4ht6QFMB0Sp'

# Input here the limiting coordinates of the extract you want to look at. You can also change the resolution.
step_x = yaml_p['size_x']/2*yaml_p['unit_xy']
step_y = yaml_p['size_y']/2*yaml_p['unit_xy']

center_lat = 46.803198
center_lon = 8.18615665

start_lat, start_lon = step(center_lat, center_lon, -step_x, -step_y)
end_lat, end_lon = step(center_lat, center_lon, step_x, step_y)
res_lat = (end_lat - start_lat)/yaml_p['size_y']
res_lon = (end_lon - start_lon)/yaml_p['size_x']

# their orgin is top left while mine was on the bottom left. Also, I have to subtract one res_lat/lon because otherwise the arrays don't match up
start_lat, end_lat = max(start_lat, end_lat-res_lat), min(start_lat, end_lat-res_lat)
start_lon, end_lon = min(start_lon, end_lon-res_lat), max(start_lon, end_lon-res_lat)

# Choose the parameter you want to get. You can only chose one parameter at a time. Check here which parameters are available: https://www.meteomatics.com/en/api/available-parameters/
dimension_list = ['elevation', 'u', 'v', 'w']
height_list = np.arange(2,3002,30.48) #data is only available starting at 2 meters

time_list = [dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)]
for _ in range(2): #add two hours to the list of times that I want to read out
    time_list.append(time_list[-1] + dt.timedelta(hours=1))

# Make placeholder for tensor
world = np.zeros(shape=(1+4,yaml_p['size_x'],yaml_p['size_y'],yaml_p['size_z']))

# Input here the date and the time
for t in time_list:
    for d in range(len(dimension_list)):
        for h in range(len(height_list)):
            try:
                if d == 0:
                    if h == 0:
                        parameter_grid = dimension_list[d] + ':m'
                        world[d,:,:,0] = api.query_grid(t, parameter_grid, start_lat, start_lon, end_lat, end_lon, res_lat, res_lon, username, password)/yaml_p['unit_z']
                        print('terrain loaded')
                        time.sleep(1.5)
                else:
                    parameter_grid = 'wind_speed_' + dimension_list[d] + '_' + str(int(height_list[h])) + 'm:ms'
                    data = api.query_grid(t, parameter_grid, start_lat, start_lon, end_lat, end_lon, res_lat, res_lon, username, password)
                    data = data.to_numpy()
                    alt = (world[0,:,:,0]+h)
                    for i in range(len(data)):
                        for j in range(len(data[0])):
                            if h == 2: #fill up lower values
                                world[d,i,j,0:int(alt[i,j])] = data[i,j]
                            if int(alt[i,j]) < len(world[0,0,0,:]): #fill up values only if within array
                                world[d,i,j,int(alt[i,j])] = data[i,j]
                                print('wind direction ' + dimension_list[d] + ' at height ' + str(int(h*yaml_p['unit_z'])) + ' m above ground')
                            else:
                                print('skipping this because out of bound')
                    time.sleep(1.5)
            except Exception as e:
                print("Failed, the exception is {}".format(e))
                time.sleep(1.5)

    train_or_test = 'test'

    digits = 4
    name_lat = str(int(np.round(center_lat,digits)*10**digits)).zfill(digits+2)
    name_lon = str(int(np.round(center_lon,digits)*10**digits)).zfill(digits+2)
    name_lat = name_lat[0:2] + '.' + name_lat[2::]
    name_lon = name_lon[0:2] + '.' + name_lon[2::]
    name = name_lat + '_' + name_lon + '_' + str(0) + '_' + str(t.hour).zfill(2) + '.pt'
    torch.save(world, '../' + yaml_p['data_path'] + train_or_test + '/tensor/' + name)

print('Current Query User Limits:')
print(api.query_user_limits(username, password))
