import numpy as np
import time
import os
import time
import json

from build_ll_controller import ll_controler
from utils.ekf import ekf

from utils.raspberrypi_com import raspi_com
from utils.raspberrypi_esc import raspi_esc
from utils.raspberrypi_gps import raspi_gps

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

def update_est(position,u,c,delta_t):
    est_x.one_cycle(0,position[0],c,delta_t)
    est_y.one_cycle(0,position[1],c,delta_t)
    est_z.one_cycle(u,position[2],c,delta_t)
    position_est = [est_x.xhat_0[0], est_y.xhat_0[0], est_z.xhat_0[0]]
    return position_est

def dist_xy(lat_1, lon_1, lat_2, lon_2):
    d_x = dist(lat_1,lon_1,lat_2,lon_1)
    d_y = dist(lat_1,lon_1,lat_1,lon_2)
    return np.array([d_x, d_y])

def dist(lat_1, lon_1, lat_2, lon_2):
    R = 6371*1000 #radius of earth in meters
    phi = lat_2 - lat_1
    lam = lon_2 - lon_1
    a = np.sin(phi/2)**2 + np.cos(lat_1) * np.cos(lat_2) * np.sin(lam/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    return d

def gps_to_position(lat,lon,height, lat_start,lon_start):
    if yaml_p['start_test'] == 'center_determ':
        offset_x = (yaml_p['size_x']-1)/2
        offset_y = (yaml_p['size_y']-1)/2
        p_x, p_y = dist_xy(lat,lon,lat_start,lon_start)/yaml_p['unit_xy'] + np.array([offset_x, offset_y])
        return [p_x, p_y, height]
    else:
        print('ERROR: please use start_test = "center" when testing')

com = raspi_com()
esc = raspi_esc()
gps = raspi_gps()

lat_start,lon_start,height_start = gps.get_gps_position()
position_gps = gps_to_position(lat_start,lon_start,height_start,lat_start,lon_start)

est_x = ekf(position_gps[0])
est_y = ekf(position_gps[1])
est_z = ekf(position_gps[2])
velocity_est = [est_x.xhat_0[1], est_y.xhat_0[1], est_z.xhat_0[1]]

offset = 0
scale = 0.1

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
while True:
    t_start = time.time()

    lat,lon,height = gps.get_gps_position()
    position_gps = gps_to_position(lat,lon,height,lat_start,lon_start)
    print('u: ' + str(u))
    position_est = update_est(position_gps,u,c,delta_t) #uses an old action for position estimation, because first estimation and then action
    velocity_est = [est_x.xhat_0[1], est_y.xhat_0[1], est_z.xhat_0[1]]

    print(position_gps)
    print('--')
    print(position_est)
    print(velocity_est)
    print('--------')

    if not os.path.isfile(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/communication/action.txt'):
        time.sleep(1)
        print('waiting for the algorithm to publish')

        data = {
        'U': U,
        'position': np.divide(position_est,[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']]).tolist(),
        'velocity': np.divide(velocity_est,[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']]).tolist(),
        'path': [],
        'position_est': np.divide(position_est,[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']]).tolist(),
        'path_est': [],
        'measurement': [0, 0],
        'min_proj_dist': 0,
        'not_done': not_done}

    else:
        data = receive()
        action = data['action']
        target = data['target']
        ceiling = data['ceiling']
        c = data['c']

        terrain = 0

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

        #action = tuning(time.time() - global_start)
        u_raw = llc.pid(action, rel_pos_est, rel_vel_est)
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

    send(data)
    if (not not_done) | (action < 0):
        break
