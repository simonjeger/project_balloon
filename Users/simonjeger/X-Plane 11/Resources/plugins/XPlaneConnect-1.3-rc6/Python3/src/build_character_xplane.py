import numpy as np
import torch
import time
import warnings
import xpc
import os
import copy
import random
from random import gauss

from build_ll_controller import ll_controler

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)
min_speed = 0
max_speed = 0

class character_xplane():
    def __init__(self, size_x, size_y, size_z, start, target, radius_xy, radius_z, T, world, world_compressed):
        self.render_ratio = yaml_p['unit_xy'] / yaml_p['unit_z']
        self.radius_xy = radius_xy
        self.radius_z = radius_z

        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.start = start.astype(float)
        self.target = target.astype(float)

        self.ll_controler = ll_controler()

        self.box_size = int(self.size_z/yaml_p['bottleneck'])

        # details about the balloon
        self.ascent_consumption = 2.5 #5 #W
        self.descent_consumption = 2.5 #W
        self.rest_consumption = 0.5 #W
        self.battery_capacity = 13187 #Ws

        self.t = T
        self.battery_level = 1
        self.action = 1

        self.world = world
        self.world_compressed = world_compressed

        self.position = copy.copy(self.start)
        self.velocity = np.array([0,0,0])
        self.n = int(yaml_p['delta_t']*1/0.5) #physics every 1/x seconds
        self.delta_tn = yaml_p['delta_t']/self.n

        self.seed = 0
        self.set_ceiling()

        self.residual = self.target - self.position
        self.measurement = [0,0,0] #TOCORRECT

        self.importance = None

        self.set_xplane()

        self.path = [self.position.copy(), self.position.copy()]
        self.min_proj_dist = np.inf
        self.min_proj_dist = np.sqrt((self.residual[0]*self.render_ratio/self.radius_xy)**2 + (self.residual[1]*self.render_ratio/self.radius_xy)**2 + (self.residual[2]/self.radius_z)**2)

        self.set_state()

    def set_xplane(self):
        #with xpc.XPlaneConnect() as self.client:
        self.client = xpc.XPlaneConnect()
        # Verify connection
        try:
            # If X-Plane does not respond to the request, a timeout error
            # will be raised.
            self.client.getDREF("sim/test/test_float")
            self.client.socket.settimeout(5) #that's totally fine for a balloon
        except:
            print("Error establishing connection to X-Plane.")
            print("Exiting...")
            return

        # Set position of the player aircraft
        #       Lat     Lon         Alt   Pitch Roll Yaw Gear
        #self.start_global = [51.5033, -0.12010, 100, 0,    0,   0,  0] #London
        #self.start_global = [47.3769, 8.5417, 500, 0,    0,   0,  0] #Zuerich
        self.start_global = [46.978715, 7.129092, 430.5, 0,    0,   0,  0] #Bellchasse
        self.target_global = [46.920049, 7.059215, 429] #Murtensee
        self.client.sendPOSI(self.start_global)

        self.old_pos_grid = [self.size_x/2, self.size_y/2,self.start_global[2]/yaml_p['unit_z']]

    def update(self, action, world_compressed, roll_out=False):
        self.action = action
        self.world_compressed = world_compressed

        not_done = self.move_particle(roll_out)
        self.set_state()

        return not_done

    def move_particle(self, roll_out):
        self.U = 0
        not_done = True
        for _ in range(self.n):
            pos = self.client.getPOSI()[0:3]
            ceiling = self.ceiling/self.size_z*3000
            terrain = pos[2] - self.client.getDREFs(['sim/flightmodel/position/y_agl'])[0][0] #m
            pos_z_squished = (pos[2] - terrain)/(ceiling - terrain)

            self.t -= yaml_p['delta_t']/self.n
            u = self.ll_controler.bangbang(self.action,pos_z_squished)

            self.U += abs(u)/self.n

            # write down path in history
            self.path.append(self.position.copy()) #because without copy otherwise it somehow overwrites it

            # find min_proj_dist
            self.residual = self.target - self.position
            min_proj_dist = np.sqrt((self.residual[0]*self.render_ratio/self.radius_xy)**2 + (self.residual[1]*self.render_ratio/self.radius_xy)**2 + (self.residual[2]/self.radius_z)**2)
            if min_proj_dist < self.min_proj_dist:
                self.min_proj_dist = min_proj_dist

            # update battery_level
            self.battery_level -= (self.rest_consumption*self.delta_tn + abs(min(u,0))*self.descent_consumption*self.delta_tn + max(u,0)*self.ascent_consumption*self.delta_tn)/self.battery_capacity

            # check if done or not
            if (self.position[0] < 0) | (self.position[0] > self.size_x):
                not_done = False
            if (self.position[1] < 0) | (self.position[1] > self.size_y):
                not_done = False
            if (pos_z_squished < 0) | (pos_z_squished > 1):
                not_done = False
            if self.t < 0: #check if flight time is over
                not_done = False
            if self.battery_level < 0: #check if battery is empty
                not_done = False

            # Send input
            dref = 'sim/flightmodel/misc/displace_rat'
            delta_disp = 0.3
            self.client.sendDREFs([dref], [1+u*delta_disp])

            # print
            os.system('clear')
            print('--- altitude control ----')
            print('current: ' + str(np.round(pos_z_squished,4)) + ', setpoint: ' + str(np.round(self.action,4)))
            print('\n')

            print('--- important information ----')
            print('remaining time: ' + str(np.round(self.t,1)) + ', battery_level: ' + str(np.round(self.battery_level,2)))
            print('\n')

            print('----- state -----')
            print('residual: ' + str(self.state[0:3]))
            print('velocity: ' + str(self.state[3:6]))
            print('boundaries: ' + str(self.state[6:10]))
            print('measurement: ' + str(self.state[10:13]))
            print('wind model: ' + str(self.state[13::]))
            print('\n')

            time.sleep(yaml_p['delta_t']/self.n)

        return not_done

    def set_state(self):
        start = self.start_global
        target = self.target_global
        # Get position and velocity
        pos = self.client.getPOSI()[0:3] #lat, long, MSL[m]
        pos_grid = self.latlongz_to_xyz(start[0], start[1], start[2], pos[0], pos[1], pos[2])/[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']] + [self.size_x/2, self.size_y/2, 0]
        res_grid = self.latlongz_to_xyz(pos[0], pos[1], pos[2], target[0], target[1], target[2])/[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']]
        self.position = pos_grid
        self.residual = res_grid

        vel = np.subtract(pos_grid, self.old_pos_grid)*[yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']] / yaml_p['delta_t']
        self.old_pos_grid = pos_grid
        self.velocity = vel

        # Boundaries
        ceiling = self.ceiling/self.size_z*3000
        terrain = pos[2] - self.client.getDREFs(['sim/flightmodel/position/y_agl'])[0][0] #m
        pos_z_squished = (pos[2] - terrain)/(ceiling - terrain)
        target_z_squished = (target[2] - start[2])/(ceiling - start[2])
        res_z_squished = target_z_squished - pos_z_squished #TOCORRECT: this assumption is that terrain at target is the same as at the start
        self.res_z_squished = res_z_squished

        total_z = (ceiling-(pos[2] - self.client.getDREFs(['sim/flightmodel/position/y_agl'])[0][0]))/ceiling
        boundaries = np.array([pos_z_squished, total_z, pos_grid[0] - np.floor(pos_grid[0]), pos_grid[1] - np.floor(pos_grid[1])])
        self.boundaries = boundaries

        # Get local wind measurement
        dir_meas = (self.client.getDREFs(['sim/weather/wind_direction_degt'])[0][0] + 180)/360*2*np.pi#rad
        mag_meas = self.client.getDREFs(['sim/weather/wind_speed_kt'])[0][0] #this is in m/s for some reason

        w_x_meas = np.sin(dir_meas)*mag_meas
        w_y_meas = np.cos(dir_meas)*mag_meas
        measurement = [w_x_meas, w_y_meas, 0]
        self.measurement = measurement

        # Get global wind model
        alt = []
        dir = []
        mag = []
        w_x = []
        w_y = []
        w_z = []

        for i in range(3):
            alt.append(self.client.getDREFs(['sim/weather/wind_altitude_msl_m[' + str(i) + ']'])[0][0]) #MSL[m]
            dir.append((self.client.getDREFs(['sim/weather/wind_direction_degt[' + str(i) + ']'])[0][0] + 180)/360*2*np.pi) #rad
            mag.append(self.client.getDREFs(['sim/weather/wind_speed_kt[' + str(i) + ']'])[0][0]*0.514444) #knots converted to m/s

            w_x.append(np.sin(dir[i])*mag[i])
            w_y.append(np.cos(dir[i])*mag[i])

        # compress wind information
        x = np.linspace(terrain,ceiling,self.size_z)
        mean_x = np.interp(x,alt,w_x)
        mean_y = np.interp(x,alt,w_y)
        mean_z = x*0

        idx = np.arange(0,self.size_z,self.box_size)
        if self.size_z%self.box_size != 0:
            idx = idx[:-1]
        pred = np.zeros((len(idx)*2)) # two different wind directions

        for i in range(len(idx)):
            with warnings.catch_warnings(): #I expect to see RuntimeWarnings in this block
                warnings.simplefilter("ignore", category=RuntimeWarning)

                pred[0*len(idx)+i] = np.nanmean(mean_x[idx[i]:idx[i] + self.box_size])
                pred[1*len(idx)+i] = np.nanmean(mean_y[idx[i]:idx[i] + self.box_size])

        pred = torch.tensor(np.nan_to_num(pred,0))
        self.world_compressed = pred

        if not yaml_p['wind_info']:
            self.world_compressed *= 0

        if not yaml_p['measurement_info']:
            self.measurement *= 0

        self.state = np.concatenate(((self.residual[0:2]/[self.size_x,self.size_y]).flatten(),[self.res_z_squished], self.normalize(self.velocity).flatten(), self.boundaries.flatten(), self.normalize(self.measurement).flatten(), self.normalize(self.world_compressed).flatten()), axis=0)

        self.bottleneck = len(self.state)
        self.state = self.state.astype(np.float32)

        if self.importance is not None:
            self.state[self.importance] = np.random.uniform(-1,1)

    def latlongz_to_xyz(self, lat1, lon1, z1, lat2, lon2, z2):
        x = self.haversine(lat1, lon1, lat1, lon2)
        y = self.haversine(lat1, lon1, lat2, lon1)
        if lon1 > lon2:
            x *= -1
        if lat1 > lat2:
            y *= -1
        z = z2 - z1
        return np.array([x,y,z])

    def haversine(self, lat1, lon1, lat2, lon2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

        c = 2 * np.arcsin(np.sqrt(a))
        d = 6367000 * c
        return d

    def set_ceiling(self):
        random.seed(self.seed)
        self.seed +=1
        self.ceiling = random.uniform(0.9, 1) * self.size_z

    def normalize(self, x):
        x = np.array(x)
        return x/(abs(x)+3)
