import numpy as np
import time
import os
import time
import json
import torch
import scipy.interpolate
import sys

from build_ll_controller import ll_controler
from utils.ekf import ekf

from utils.raspberrypi_com import raspi_com
from utils.raspberrypi_esc import raspi_esc
from utils.raspberrypi_gps import raspi_gps
from utils.raspberrypi_alt import raspi_alt

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

def send(data):
    path = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/communication/'
    with open(path + 'data.txt', 'w') as f:
        f.write(json.dumps(data))
    return data

def receive():
    successful = False
    path = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/communication/'
    t_start = time.time()
    corrupt = False
    while not successful:
        with open(path + 'action.txt') as json_file:
            try:
                data = json.load(json_file)
                successful = True
            except:
                corrupt = True
    if corrupt:
        print('data corrupted, lag of ' + str(np.round(time.time() - t_start,3)) + '[s]')
    return data

def update_est(position,u,c,delta_t,delta_f_up,delta_f_down,mass_total):
    force_est = (max(0,u)*delta_f_up + min(0,u)*delta_f_down)/yaml_p['unit_z']/mass_total
    est_x.one_cycle(0,position[0],c,delta_t)
    est_y.one_cycle(0,position[1],c,delta_t)
    est_z.one_cycle(force_est,position[2],c,delta_t)
    position_est = [est_x.xhat_0[0], est_y.xhat_0[0], est_z.xhat_0[0]]
    return position_est

def dist_xy(lat_1, lon_1, lat_2, lon_2):
    d_x = dist(lat_1,lon_1,lat_2,lon_1)
    d_y = dist(lat_1,lon_1,lat_1,lon_2)
    return np.array([d_x, d_y])

def dist(lat_1, lon_1, lat_2, lon_2):
    lat_1 = np.radians(lat_1)
    lon_1 = np.radians(lon_1)
    lat_2 = np.radians(lat_2)
    lon_2 = np.radians(lon_2)

    R = 6371*1000 #radius of earth in meters
    phi = lat_2 - lat_1
    lam = lon_2 - lon_1
    a = np.sin(phi/2)**2 + np.cos(lat_1) * np.cos(lat_2) * np.sin(lam/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    return d

def gps_to_position(lat,lon,height, lat_start,lon_start):
    # lat_start and lon_start where measured in the middle of the map
    offset_x = (yaml_p['size_x']-1)/2
    offset_y = (yaml_p['size_y']-1)/2
    p_x, p_y = dist_xy(lat,lon,lat_start,lon_start)/yaml_p['unit_xy'] + np.array([offset_x, offset_y])
    return [p_x, p_y, height/yaml_p['unit_z']]

def get_center():
    center = torch.load(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/render/coord.pt')
    return center[0], center[1]

# interpolation for terrain
world = torch.load(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/render/world.pt')
x = np.linspace(0,len(world[0,:,0,0]),len(world[0,:,0,0]))
y = np.linspace(0,len(world[0,0,:,0]),len(world[0,0,:,0]))
f_terrain = scipy.interpolate.interp2d(x,y,world[0,:,:,0].T)

# initialize devices
#com = raspi_com()
esc = raspi_esc()
gps = raspi_gps()
alt = raspi_alt()

lat_start,lon_start = get_center()
lat,lon,height = gps.get_gps_position()
position_meas = gps_to_position(lat,lon,height,lat_start,lon_start)
position_meas[2] = alt.get_altitude()/yaml_p['unit_z']

est_x = ekf(position_meas[0])
est_y = ekf(position_meas[1])
est_z = ekf(position_meas[2])

velocity_est = [est_x.xhat_0[1], est_y.xhat_0[1], est_z.xhat_0[1]]

offset = yaml_p['offset']
scale = yaml_p['scale']

llc = ll_controler()

path = []
path_est = []
not_done = True

U = 0
u = 0
min_proj_dist = np.inf

c = 1 #only placeholder, nescessary for estimation functions
delta_t = 20 #only placeholder, nescessary for estimation functions

global_start = time.time()
print('System ready')
while True:
    t_start = time.time()

    # load center of map (depending on when the agent starts this needs to be rechecked)
    center = torch.load(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/render/coord.pt')
    lat_start,lon_start = get_center()

    if not os.path.isfile(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/communication/action.txt'):
        time.sleep(1)
        print('waiting for the algorithm to publish')

        data = {
        'U': U,
        'position': position_meas,
        'velocity': [0,0,0],
        'path': [],
        'position_est': position_meas,
        'path_est': [],
        'measurement': [0, 0],
        'min_proj_dist': 0,
        'not_done': not_done}

        action = 0 #there is no command to reveive anything from, so I assume the action = 0 (only important for break out condition at the end)

    else:
        try: #if anything breaks, just cut the motors
            data = receive()
            action = data['action']
            target = data['target']
            ceiling = data['ceiling']
            c = data['c']
            delta_f_up = data['delta_f_up']
            delta_f_down = data['delta_f_down']
            mass_total = data['mass_total']

            #get GPS data and use it
            try:
                lat,lon,height = gps.get_gps_position()
            except:
                time.sleep(1) #that's usually about as long as it takes for a measurement to get in
                print("WARNING: Couldn't get GPS measurement at " + str(int(t_start - global_start)) + 's after start.')
            position_meas = gps_to_position(lat,lon,height,lat_start,lon_start)
            try:
                position_meas[2] = alt.get_altitude()
            except:
                print("WARNING: Couldn't get ALT measurement at " + str(int(t_start - global_start)) + 's after start.')
            position_est = update_est(position_meas,u,c,delta_t,delta_f_up,delta_f_down,mass_total) #uses an old action for position estimation, because first estimation and then action
            velocity_est = [est_x.xhat_0[1], est_y.xhat_0[1], est_z.xhat_0[1]]
            terrain = f_terrain(position_est[0], position_est[1])[0]

            # degbug
            #est_y.plot()

            rel_pos_est = (position_est[2] - terrain)/(ceiling-terrain)
            rel_vel_est = velocity_est[2] / (ceiling-terrain)

            # check if done or not
            if (position_est[0] < 0) | (position_est[0]/yaml_p['unit_xy'] > yaml_p['size_x'] - 1):
                print('x out of bounds')
                not_done = False
            if (position_est[1] < 0) | (position_est[1]/yaml_p['unit_xy'] > yaml_p['size_y'] - 1):
                print('y out of bounds')
                not_done = False
            """
            if (rel_pos_est < 0) | (rel_pos_est >= 1):
                print('z out of bounds')
                not_done = False
            if t < 0: #check if flight time is over
                not_done = False
            if self.battery_level < 0: #check if battery is empty
                not_done = False
            """

            u_raw = llc.bangbang(action, rel_pos_est)
            u = offset + u_raw*scale

            if yaml_p['mode'] == 'tuning':
                print(u)

            esc.control(u)
            if (not not_done) | (action < 0):
                u = 0
                esc.control(u)

            t_stop = time.time()
            delta_t = t_stop - t_start

            path_est.append(np.divide(position_est,[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']]).tolist())

            U += abs(u*delta_t)

            # find min_proj_dist
            render_ratio = yaml_p['unit_xy']/yaml_p['unit_z']
            residual = target - np.divide(position_est,[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']])
            min_proj_dist_prop = np.sqrt((residual[1]*render_ratio/yaml_p['radius_xy'])**2 + (residual[2]/yaml_p['radius_z'])**2) #only 2d case!
            min_dist_prop = np.sqrt((residual[1]*render_ratio)**2 + (residual[2])**2)*yaml_p['unit_z']
            if min_proj_dist_prop < min_proj_dist:
                min_proj_dist = min_proj_dist_prop
                min_dist = min_dist_prop

            data = {
            'U': U,
            'position': np.divide(position_est,[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']]).tolist(),
            'velocity': np.divide(velocity_est,[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']]).tolist(),
            'path': path_est,
            'position_est': position_est,
            'path_est': path_est,
            'measurement': [est_x.wind(), est_y.wind()],
            'min_proj_dist': min_proj_dist,
            'min_dist': min_dist,
            'not_done': not_done}

            if not not_done:
                print('INFORMATION: system ran out of bounds.')
                break
            if action < 0:
                print('INFORMATION: run was cancalled deliberately.')
                break

        except KeyboardInterrupt:
            print("WARNING: Maual kill.")
            sys.exit()

        except:
            print("WARNING: Something fatal broke down at " + str(int(global_start - t_start)) + 's after start.')
            u = 0
            esc.control(u)
            print("INFORMATION: Thrust set to zero.")
