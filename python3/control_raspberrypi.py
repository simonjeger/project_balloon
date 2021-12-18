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

def send(data):
    path = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/communication/'
    with open(path + 'data.txt', 'w') as f:
        f.write(json.dumps(data))
    return data

def receive():
    successful = False
    path = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/communication/'
    start = time.time()
    corrupt = False
    while not successful:
        with open(path + 'action.txt') as json_file:
            try:
                data = json.load(json_file)
                successful = True
            except:
                corrupt = True
    if corrupt:
        print('data corrupted, lag of ' + str(np.round(time.time() - start,3)) + '[s]')
    return data

com = raspi_com()
esc = raspi_esc()
gps = raspi_gps()

position_gps = gps.get_gps_position()
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

plot_time = []
plot_pos_x = []
plot_pos_y = []
plot_pos_z = []
plot_vel = []
plot_u = []
plot_action = []
plot_residual = np.inf

U = 0
u = 0
min_proj_dist = np.inf

global_start = time.time()
while True:
    start = time.time()

    position_gps = gps.get_gps_position()
    velocity = [est_x.xhat_0[1], est_y.xhat_0[1], est_z.xhat_0[1]]
    position_est = np.divide(update_est(position_gps,u,c,delta_t),[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']]) #uses an old action for position estimation, because first estimation and then action

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
        if (rel_pos_est < 0) | (rel_pos_est >= 1):
            print('z out of bounds')
            not_done = False
        if self.t < 0: #check if flight time is over
            not_done = False
        if self.battery_level < 0: #check if battery is empty
            not_done = False

        #action = tuning(time.time() - global_start)
        u_raw = llc.pid(action, rel_pos, rel_vel)
        u = offset + u_raw*scale

        if yaml_p['mode'] == 'tuning':
            print(u)

        esc.control(u)
        if (not not_done) | (action < 0):
            u = 0
            esc.control(u)
            plot_pid(plot_time, plot_pos_z, plot_vel, plot_u, plot_action)

        stop = time.time()
        delta_t = stop - start

        path_est.append(np.divide(position_est,[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']]).tolist())

        plot_pos_x.append(position_est[0])
        plot_pos_y.append(position_est[1])
        plot_pos_z.append(position_est[2])
        plot_vel.append(velocity_est[2])
        plot_u.append(u)
        plot_action.append(action*ceiling)
        plot_time.append(time.time())

        U += abs(u*delta_t)

        # find min_proj_dist
        render_ratio = yaml_p['unit_xy']/yaml_p['unit_z']
        residual = target - np.divide(position_est,[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']])
        min_proj_dist_prop = np.sqrt((residual[1]*render_ratio/yaml_p['radius_xy'])**2 + (residual[2]/yaml_p['radius_z'])**2) #only 2d case!
        min_dist_prop = np.sqrt((residual[1]*render_ratio)**2 + (residual[2])**2)*yaml_p['unit_z']
        if min_proj_dist_prop < min_proj_dist:
            min_proj_dist = min_proj_dist_prop
            min_dist = min_dist_prop
            plot_residual = target*np.array([yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']]) - position_est

        data = {
        'U': U,
        'position': np.divide(position_est,[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']]).tolist(),
        'velocity': np.divide(velocity_est,[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']]).tolist(),
        'path': path_est,
        'position_est': position_est.tolist(),
        'path_est': path_est,
        'measurement': [est_x.wind(), est_y.wind()],
        'min_proj_dist': min_proj_dist,
        'min_dist': min_dist,
        'not_done': not_done}

    send(data)
    if (not not_done) | (action < 0):
        break
