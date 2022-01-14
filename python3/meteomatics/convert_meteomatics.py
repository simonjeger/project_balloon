import datetime as dt
import meteomatics.api as api
import numpy as np
import time
import torch
import datetime

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

center_lat = 46.803198
center_lon = 8.18615665

start_lat, start_lon = step(center_lat, center_lon, -step_x, -step_y)
end_lat, end_lon = step(center_lat, center_lon, step_x, step_y)
res_lat = abs(end_lat - start_lat)/yaml_p['size_y']
res_lon = abs(end_lon - start_lon)/yaml_p['size_x']

# their orgin is top left while mine was on the bottom left. Also, I have to subtract one res_lat/lon because otherwise the arrays don't match up
start_lat, end_lat = max(start_lat, end_lat-res_lat), min(start_lat, end_lat-res_lat)
start_lon, end_lon = min(start_lon, end_lon-res_lat), max(start_lon, end_lon-res_lat)

# their function can only handle precision up to 6 digits, otherwise it will throw an error
start_lat = np.round(start_lat,6)
start_lon = np.round(start_lon,6)
end_lat = np.round(end_lat,6)
end_lon = np.round(end_lon,6)
res_lat = np.round(res_lat,6)
res_lon = np.round(res_lon,6)

N_time = int(np.ceil(yaml_p['T']/60/60))
start_date = datetime.datetime.today()
end_date = datetime.datetime.today() + datetime.timedelta(hours=N_time)
res_time = datetime.timedelta(hours=1)

height = np.arange(2,3002,30.48) #data is only available starting at 2 meters

# Make placeholder
elevation = []
wind_speed_u = []
wind_speed_v = []

# Input here the date and the time
for h in height:
    #try:
    parameter_grid = ['elevation:m', 'wind_speed_u_' + str(int(h)) + 'm:ms', 'wind_speed_v_' + str(int(h)) + 'm:ms']
    request = api.query_grid_timeseries(start_date, end_date, res_time, parameter_grid, start_lat, start_lon, end_lat, end_lon, res_lat, res_lon, username, password)

    elevation.append(request['elevation:m']) #even tho we just need it the first time, it's easier to keep it in the loop and only use the first entry
    wind_speed_u.append(request['wind_speed_u_' + str(int(h)) + 'm:ms'])
    wind_speed_v.append(request['wind_speed_v_' + str(int(h)) + 'm:ms'])

    time.sleep(1.5)
    #except Exception as e:
    #    print("Failed, the exception is {}".format(e))
    #    time.sleep(1.5)

print('Current Query User Limits:')
print(api.query_user_limits(username, password))

world = np.zeros(shape=(1+4,yaml_p['size_x'],yaml_p['size_y'],yaml_p['size_z']))
world[0,:,:,0] = elevation[0]
for t in range(N_time):
    for i in range(yaml_p['size_x']):
        #for j in range(yaml_p['size_y']):
            #for k in range(yaml_p['size_z']):

    train_or_test = 'test'
    digits = 4
    name_lat = str(int(np.round(center_lat,digits)*10**digits)).zfill(digits+2)
    name_lon = str(int(np.round(center_lon,digits)*10**digits)).zfill(digits+2)
    name_lat = name_lat[0:2] + '.' + name_lat[2::]
    name_lon = name_lon[0:2] + '.' + name_lon[2::]
    name = name_lat + '_' + name_lon + '_' + str(0) + '_' + str(start_date.hour + t).zfill(2) + '.pt'
    torch.save(world, '../' + yaml_p['data_path'] + train_or_test + '/tensor/' + name)

print('Current Query User Limits:')
print(api.query_user_limits(username, password))
