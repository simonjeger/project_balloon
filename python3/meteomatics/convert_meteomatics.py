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

# Credentials:
username = 'jungfraubergbahnen_zimmer'
password = '4dCzy38UQsWlK'

# Input here the limiting coordinates of the extract you want to look at. You can also change the resolution.
step_x = yaml_p['size_x']/2*yaml_p['unit_xy']
step_y = yaml_p['size_y']/2*yaml_p['unit_xy']

center_lat = yaml_p['center_latlon'][0]
center_lon = yaml_p['center_latlon'][1]

start_lat, start_lon = step(center_lat, center_lon, -step_x, -step_y)
end_lat, end_lon = step(center_lat, center_lon, step_x, step_y)
res_lat = abs(end_lat - start_lat)/yaml_p['size_y']
res_lon = abs(end_lon - start_lon)/yaml_p['size_x']

# their orgin is top left while mine was on the bottom left
start_lat, end_lat = max(start_lat, end_lat), min(start_lat, end_lat)
start_lon, end_lon = min(start_lon, end_lon), max(start_lon, end_lon)

# their function can only handle precision up to 6 digits, otherwise it will throw an error
start_lat = np.round(start_lat,6)
start_lon = np.round(start_lon,6)
end_lat = np.round(end_lat,6)
end_lon = np.round(end_lon,6)
res_lat = np.ceil(res_lat*1e6)/1e6 #I have to round up, otherwise the array below has to many entries
res_lon = np.ceil(res_lon*1e6)/1e6

N_time = int(np.ceil(yaml_p['T']/60/60)) + 1
now = datetime.datetime.today()
start_date = datetime.datetime(now.year, now.month, now.day)
end_date = datetime.datetime(now.year, now.month, now.day) + datetime.timedelta(hours=23,minutes=59,seconds=59)
res_time = datetime.timedelta(hours=1)

height = np.arange(2,yaml_p['size_z']*yaml_p['unit_z']+2,yaml_p['unit_z']) #data is only available starting at 2 meters

# Make placeholder
elevation = []
wind_speed_u = []
wind_speed_v = []

print('Current Query User Limit Status:')
print(api.query_user_limits(username, password))

# Input here the date and the time
h = 0
while h < len(height):
    parameter_grid = []
    while True:
        if h == 0:
            parameter_grid.append('elevation:m')
        parameter_grid.append('wind_speed_u_' + str(int(height[h])) + 'm:ms')
        parameter_grid.append('wind_speed_v_' + str(int(height[h])) + 'm:ms')
        h += 1
        if (len(parameter_grid) >= 9) | (h >= len(height) - 1):
            break

    request = api.query_grid_timeseries(start_date, end_date, res_time, parameter_grid, start_lat, start_lon, end_lat, end_lon, res_lat, res_lon, username, password)
    time.sleep(1.5)

    for p in range(len(height)):
        try:
            elevation.append(request['elevation:m'])
        except:
            pass
        try:
            wind_speed_u.append(request['wind_speed_u_' + str(int(height[p])) + 'm:ms'])
        except:
            pass
        try:
            wind_speed_v.append(request['wind_speed_v_' + str(int(height[p])) + 'm:ms'])
        except:
            pass
    print('Downloaded ' + str(np.round(h/len(height)*100,1)) + '% of meteomatics data')

world = np.zeros(shape=(1+4,yaml_p['size_x'],yaml_p['size_y'],yaml_p['size_z']))

# generate directory
train_or_test = 'test'
Path('../' + yaml_p['data_path']).mkdir(parents=True, exist_ok=True)
Path('../' + yaml_p['data_path'] + train_or_test).mkdir(parents=True, exist_ok=True)
Path('../' + yaml_p['data_path'] + train_or_test + '/tensor').mkdir(parents=True, exist_ok=True)

elevation_h = elevation[0].to_numpy().reshape(yaml_p['size_y'],yaml_p['size_x'],-1)
elevation_ht = np.swapaxes(elevation_h[:,:,0],0,1) #the tensors are coord_y, coord_x, time
world[0,:,:,0] = elevation_ht/yaml_p['unit_z']

for t in range(len(elevation_h[0,0,:])):
    for k in range(yaml_p['size_z']):
        wind_speed_u_h = wind_speed_u[k].to_numpy().reshape(yaml_p['size_y'],yaml_p['size_x'],-1)
        wind_speed_u_ht = np.swapaxes(wind_speed_u_h[:,:,t],0,1) #the tensors are coord_y, coord_x, time
        wind_speed_v_h = wind_speed_v[k].to_numpy().reshape(yaml_p['size_y'],yaml_p['size_x'],-1)
        wind_speed_v_ht = np.swapaxes(wind_speed_v_h[:,:,t],0,1) #the tensors are coord_y, coord_x, time

        for i in range(yaml_p['size_x']):
            for j in range(yaml_p['size_y']):
                alt = int(world[0,i,j,0] + k)
                if k == 0:
                    world[1,i,j,0:alt] = wind_speed_u_ht[i,j]
                    world[2,i,j,0:alt] = wind_speed_v_ht[i,j]
                if alt < yaml_p['size_z']:
                    world[1,i,j,alt] = wind_speed_u_ht[i,j]
                    world[2,i,j,alt] = wind_speed_v_ht[i,j]
    print('Generated ' + str(int((t)/(len(elevation_h[0,0,:]))*100)) + '% of the meteomatics tensors')

    # in every timestamp I generate a tensor
    digits = 4
    name_lat = str(int(np.round(center_lat,digits)*10**digits)).zfill(digits+2)
    name_lon = str(int(np.round(center_lon,digits)*10**digits)).zfill(digits+2)
    name_lat = name_lat[0:2] + '.' + name_lat[2::]
    name_lon = name_lon[0:2] + '.' + name_lon[2::]
    name = name_lat + '_' + name_lon + '_' + str(0) + '_' + str(start_date.hour + t).zfill(2) + '.pt'
    torch.save(world, '../' + yaml_p['data_path'] + train_or_test + '/tensor/' + name)

print('Current Query User Limit Status:')
print(api.query_user_limits(username, password))
