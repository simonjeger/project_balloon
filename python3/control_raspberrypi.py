import numpy as np
import time
import os
import time
import json
import torch
import scipy.interpolate
import sys
import shutil

from build_ll_controller import ll_controler
from utils.ekf import ekf

from utils.raspberrypi_esc import raspi_esc
from utils.raspberrypi_gps import raspi_gps
from utils.raspberrypi_alt import raspi_alt

import logging
logging.basicConfig(filename="logger/control_raspberrypi.log", format='%(asctime)s %(message)s', filemode='w')
logging.getLogger().addHandler(logging.StreamHandler())
logger=logging.getLogger()
logger.setLevel(logging.INFO)

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
        logger.warning('RBP: data corrupted, lag of ' + str(np.round(time.time() - t_start,3)) + '[s]')
    return data

def update_est(position,u,c,delta_t,delta_f_up,delta_f_down,mass_total):
    force_est = (max(0,u)*delta_f_up + min(0,u)*delta_f_down)/yaml_p['unit_z']/mass_total
    est_x.one_cycle(0,position[0],c,delta_t)
    est_y.one_cycle(0,position[1],c,delta_t)
    est_z.one_cycle(force_est,position[2],c,delta_t)
    position_est = [est_x.xhat_0[0], est_y.xhat_0[0], est_z.xhat_0[0]]
    return position_est

def dist_xy(lat_1, lon_1, lat_2, lon_2):
    d_x = np.sign(lon_2 - lon_1)*dist(lat_1,lon_1,lat_1,lon_2)
    d_y = np.sign(lat_2 - lat_1)*dist(lat_1,lon_1,lat_2,lon_1)
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
    p_x, p_y = dist_xy(lat_start,lon_start,lat,lon)/yaml_p['unit_xy'] + np.array([offset_x, offset_y])
    return [p_x, p_y, height/yaml_p['unit_z']]

def get_center():
    center = torch.load(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/render/coord.pt')
    return center[0], center[1]

# clear all previous communication files
path = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/communication'
if os.path.exists(path):
    shutil.rmtree(path)
    os.makedirs(path)

# interpolation for terrain
world = torch.load(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/render/world.pt')
x = np.linspace(0,len(world[0,:,0,0]),len(world[0,:,0,0]))
y = np.linspace(0,len(world[0,0,:,0]),len(world[0,0,:,0]))
f_terrain = scipy.interpolate.interp2d(x,y,world[0,:,:,0].T)

# initialize devices
alt = raspi_alt()
lat_start,lon_start = get_center()
while True: #search until found
    try:
        gps = raspi_gps(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/communication/')
        lat,lon,height = gps.get_gps_position(max_cycles=60)
        break
    except:
        logger.error('RBP: failed to find GPS fixture, will try again')
        gps.power_off()

position_meas = gps_to_position(lat,lon,height,lat_start,lon_start)

#set the altimeter
terrain = f_terrain(position_meas[0], position_meas[1])[0]
alt.set_QNH(terrain)
position_meas[2] = alt.get_altitude()/yaml_p['unit_z']

est_x = ekf(position_meas[0])
est_y = ekf(position_meas[1])
est_z = ekf(position_meas[2])

offset = yaml_p['offset']
scale = yaml_p['scale']

llc = ll_controler()

path = []
path_est = []
not_done = True
landed = False

U_integrated = 0
u = 0
min_proj_dist = np.inf

# only placeholder, nescessary for estimation functions
c = 1
delta_t = 2
delta_f_up = 2.5
delta_f_down = 2.5
mass_total = 1.2

position_est = update_est(position_meas,u,c,delta_t,delta_f_up,delta_f_down,mass_total)
velocity_est = [est_x.xhat_0[1], est_y.xhat_0[1], est_z.xhat_0[1]]

global_start = time.time()
esc = raspi_esc() #only arm when ready
logger.info('RBP: ready')

while True:
    t_start = time.time()

    # load center of map (depending on when the agent starts this needs to be rechecked)
    center = torch.load(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/render/coord.pt')
    lat_start,lon_start = get_center()

    if not os.path.isfile(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/communication/action.txt'):
        time.sleep(1)
        logger.info('RBP: waiting for the algorithm to publish at ' + str(int(time.time() - global_start)) + ' s after starting')

        position_est = update_est(position_meas,u,c,delta_t,delta_f_up,delta_f_down,mass_total)
        velocity_est = [est_x.xhat_0[1], est_y.xhat_0[1], est_z.xhat_0[1]]

        data = {
        'U_integrated': U_integrated,
        'position': position_est,
        'velocity': velocity_est,
        'path': path_est,
        'position_est': position_est,
        'path_est': path_est,
        'measurement': [est_x.wind(), est_y.wind()],
        'min_proj_dist': 100,
        'min_dist': 100,
        'not_done': not_done,
        'delta_t': delta_t,
        'gps_lat': lat,
        'gps_lon': lon,
        'gps_height': height,
        'rel_pos_est': 0,
        'u': u}

        send(data)

        action = 0 #there is no command to reveive anything from, so I assume the action = 0 (only important for break out condition at the end)

    else:
        try: #if anything breaks, just cut the motors
            data = receive()
            if not data['action_overwrite']:
                action = data['action']
            else:
                action = data['action_overwrite']
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
                time.sleep(2) #that's usually about as long as it takes for a measurement to get in
                logger.warning("RBP: Couldn't get GPS measurement at " + str(int(t_start - global_start)) + ' s after start.')
            position_meas = gps_to_position(lat,lon,height,lat_start,lon_start)
            try:
                position_meas[2] = alt.get_altitude()/yaml_p['unit_z']
            except:
                logger.warning("RBP: Couldn't get ALT measurement at " + str(int(t_start - global_start)) + ' s after start.')
            position_est = update_est(position_meas,u,c,delta_t,delta_f_up,delta_f_down,mass_total) #uses an old action for position estimation, because first estimation and then action
            velocity_est = [est_x.xhat_0[1], est_y.xhat_0[1], est_z.xhat_0[1]]
            terrain = f_terrain(position_est[0], position_est[1])[0]

            # degbug
            #est_y.plot()

            rel_pos_est = (position_est[2] - terrain)/(ceiling-terrain)
            rel_vel_est = velocity_est[2] / (ceiling-terrain)

            if not landed:
                # check if done or not
                if (position_est[0] < 0) | (position_est[0] > yaml_p['size_x'] - 1):
                    if not_done:
                        logger.info('RBP: X out of bounds')
                    not_done = False
                if (position_est[1] < 0) | (position_est[1] > yaml_p['size_y'] - 1):
                    if not_done:
                        logger.info('RBP: Y out of bounds')
                    not_done = False
                if t_start - global_start > yaml_p['T'] + 60: #check if flight time is over and give a minute of buffer because usually the raspi get's started earlier
                    if not_done:
                        logger.info('RBP: Out of time')
                    not_done = False
                if action < 0:
                    if not_done:
                        logger.info('RBP: run was cancalled deliberately')
                    not_done = False

                # control input u for flight case and landing
                u_raw = llc.bangbang(action, rel_pos_est)
                if not_done == False:
                    cutoff_height = 0.05
                    d_m = np.max([int((position_est[2] - ((ceiling - terrain)*cutoff_height + terrain))*yaml_p['unit_z']),0])
                    logger.info('RBP: landing with ' + str(d_m) + ' m to go until motor cut-off')
                    if rel_pos_est > cutoff_height:
                        u_raw = -1
                    else:
                        u_raw = 0
                        landed = True
                        logger.info('RBP: landed')
                u = offset + u_raw*scale
                if not landed:
                    esc.control(u)
                else:
                    u = 0 #just for the print during tuning
                    esc.stop()

            if yaml_p['mode'] == 'tuning':
                logger.debug('--- tuning ---')
                logger.debug('altitude: ' + str(alt.get_altitude()) + ' m')
                logger.debug('position_est: ' + str(position_est))
                logger.debug('rel_pos_est: ' + str(rel_pos_est))
                logger.debug('terrain: ' + str(terrain) + ' m')
                logger.debug('t: ' + str(int(t_start - global_start)) + ' s')
                logger.debug('u: ' + str(u))

            t_stop = time.time()
            delta_t = t_stop - t_start

            path_est.append(position_est)

            U_integrated += abs(u*delta_t)

            # find min_proj_dist
            render_ratio = yaml_p['unit_xy']/yaml_p['unit_z']
            residual = target - np.divide(position_est,[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']])
            min_proj_dist_prop = np.sqrt((residual[1]*render_ratio/yaml_p['radius_xy'])**2 + (residual[2]/yaml_p['radius_z'])**2) #only 2d case!
            min_dist_prop = np.sqrt((residual[1]*render_ratio)**2 + (residual[2])**2)*yaml_p['unit_z']
            if min_proj_dist_prop < min_proj_dist:
                min_proj_dist = min_proj_dist_prop
                min_dist = min_dist_prop

            data = {
            'U_integrated': U_integrated,
            'position': position_est,
            'velocity': velocity_est,
            'path': path_est,
            'position_est': position_est,
            'path_est': path_est,
            'measurement': [est_x.wind(), est_y.wind()],
            'min_proj_dist': min_proj_dist,
            'min_dist': min_dist,
            'not_done': not_done,
            'delta_t': delta_t,
            'gps_lat': lat,
            'gps_lon': lon,
            'gps_height': height,
            'rel_pos_est': rel_pos_est,
            'u': u}

            send(data)

        except KeyboardInterrupt:
            logger.info("RBP: Maual kill")
            esc.stop()
            gps.power_off()
            sys.exit()

        except:
            logger.error("RBP: Something fatal broke down at " + str(int(t_start - global_start)) + ' s after start')
            esc.stop()
            gps.power_off()
            sys.exit()
