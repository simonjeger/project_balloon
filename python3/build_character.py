import numpy as np
import scipy
import copy
import torch
import os
from human_autoencoder import HAE
from build_autoencoder import VAE
from scipy.interpolate import NearestNDInterpolator
from sklearn.neighbors import BallTree
from scipy.ndimage import gaussian_filter
from scipy.interpolate import UnivariateSpline
import time
import json

from preprocess_wind import squish
from build_ll_controller import ll_controler
from utils.ekf import ekf
from visualize_world import visualize_world #for debugging only

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

class character():
    def __init__(self, size_x, size_y, size_z, start, target, radius_xy, radius_z, T, world, train_or_test, seed):
        if train_or_test == 'train': #only testing is affected by denser logging to avoid messing up the learning
            yaml_p['delta_t_logger'] = yaml_p['delta_t']
        self.n = int(yaml_p['delta_t_logger']/yaml_p['delta_t_physics'])
        self.delta_tn = yaml_p['delta_t_logger']/self.n

        self.render_ratio = yaml_p['unit_xy'] / yaml_p['unit_z']
        self.radius_xy = radius_xy
        self.radius_z = radius_z

        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.start = np.array(start).astype(float)
        self.target = np.array(target).astype(float)

        self.position = copy.copy(self.start)
        self.velocity = np.array([0,0,0])

        self.ll_controler = ll_controler()
        self.est_x = ekf(self.position[0])
        self.est_y = ekf(self.position[1])
        self.est_z = ekf(self.position[2])

        self.esterror_pos = 0
        self.esterror_vel = 0
        self.esterror_wind = 0

        if yaml_p['balloon'] == 'outdoor_balloon':
            self.mass_structure = 0.839 + 0.247 #1.2 #kg
            self.delta_f_up = 2.5 #N
            self.delta_f_down = 2.5 #N
            self.delay = 1 #s
            self.consumption_up = 70 #W
            self.consumption_down = 70 #W
            self.rest_consumption = 1.5 #W
            self.battery_capacity = 266400 #Ws
            self.c_w = 0.795

        elif yaml_p['balloon'] == 'indoor_balloon':
            self.mass_structure = 0.049 #kg
            self.delta_f_up = 0.06 #N
            self.delta_f_down = 0.03 #N
            self.delay = 0.5 #0.6 #s
            self.consumption_up = 5 #W
            self.consumption_down = 2.5 #W
            self.rest_consumption = 0.5 #W
            self.battery_capacity = 1798 #Ws
            self.c_w = 0.6714 #through experiment
        else:
            print('ERROR: please choose one of the available balloons')

        # initialize autoencoder object
        if yaml_p['autoencoder'][0:3] == 'HAE':
            self.ae = HAE()
        if yaml_p['autoencoder'] == 'VAE':
            self.ae = VAE()
            self.ae.load_weights('autoencoder/model_' + str(yaml_p['vae_nr']) + '.pt')

        self.T = T
        self.t = self.T

        self.battery_level = 1
        self.action = 0.01
        self.action_hist = [self.action]
        self.diameter = 0

        self.world = world

        self.world_est = np.zeros_like(self.world)
        self.world_est[0] = self.world[0] #terrain is known
        self.world_est_bn = int(min(yaml_p['bottleneck']*4,self.size_z))
        self.world_est_mask = np.ones(self.world_est_bn)*-self.world_est_bn
        self.world_est_mask[-1] += 0.1 #so it detects the last entry of the array through argmin
        self.world_est_data = np.zeros((2,self.world_est_bn))

        self.w_est_xy = yaml_p['unit_xy']
        self.w_est_z = yaml_p['unit_z']*15
        self.w_est_t = yaml_p['delta_t']*0.00001

        self.train_or_test = train_or_test

        self.position_est = copy.copy(self.position)
        self.velocity_est = np.array([0,0,0])

        # interpolation for terrain
        x = np.linspace(0,self.size_x,len(self.world[0,:,0,0]))
        y = np.linspace(0,self.size_y,len(self.world[0,0,:,0]))
        self.f_terrain = scipy.interpolate.interp2d(x,y,self.world[0,:,:,0].T)

        self.seed = seed
        self.set_ceiling()
        self.world_squished = squish(self.world, self.ceiling)

        if yaml_p['environment'] == 'python3':
            self.set_noise()

        self.residual = self.target - self.position
        self.measurement = self.interpolate(self.world_squished)[0:2]

        self.importance = None
        self.adapt_volume(0) #initial volume with u=0
        self.rel_pos_est = self.height_above_ground(est=True)/(self.ceiling-(self.position_est[2]-self.height_above_ground(est=True)))
        self.set_state()

        self.path = [self.position.copy(), self.position.copy()]
        self.path_est = [self.position.copy(), self.position.copy()]

        self.min_proj_dist = np.inf
        self.min_dist = np.inf

        if yaml_p['3d']:
            self.min_proj_dist = np.sqrt((self.residual[0]*self.render_ratio/self.radius_xy)**2 + (self.residual[1]*self.render_ratio/self.radius_xy)**2 + (self.residual[2]/self.radius_z)**2)
            self.min_dist = np.sqrt((self.residual[0]*self.render_ratio)**2 + (self.residual[1]*self.render_ratio)**2 + (self.residual[2])**2)
        else:
            self.min_proj_dist = np.sqrt((self.residual[1]*self.render_ratio/self.radius_xy)**2 + (self.residual[2]/self.radius_z)**2)
            self.min_dist = np.sqrt((self.residual[1]*self.render_ratio)**2 + (self.residual[2])**2)
        #for move_particle (previous velocity is zero at the beginning)
        self.p_x = 0
        self.p_y = 0
        self.p_z = 0

        self.real_time = copy.copy(time.time())
        self.position_old = self.position
        self.position_est_old = self.position_est

    def update(self, action, world):
        self.action = action
        self.action_hist.append(action)
        self.world = world
        self.world_squished = squish(self.world, self.ceiling)

        if yaml_p['environment'] == 'python3':
            not_done = self.move_particle()
        else:
            not_done = self.live_particle()

        # update state
        self.set_state()

        #to debug
        #self.est_y.plot()

        return not_done

    def set_state(self):
        # residual
        self.residual = self.target - self.position
        self.residual_est = self.target - self.position_est

        # Update compressed wind map
        if yaml_p['world_est']:
            self.update_world_est()
            self.world_compressed = self.ae.compress_est(self.world_est, self.position_est, self.ceiling)
            if yaml_p['log_world_est_error']:
                ground_truth = self.ae.compress(self.world, self.position_est, self.ceiling)
                if np.linalg.norm(ground_truth) != 0:
                    self.esterror_world = np.linalg.norm(self.world_compressed - ground_truth)/np.linalg.norm(ground_truth)
                else:
                    self.esterror_world = np.inf

        else:
            self.world_compressed = self.ae.compress(self.world, self.position_est, self.ceiling)
        self.world_compressed /= yaml_p['unit_xy'] #so it's in simulation units and makes sense for the normalization in character.py

        if not yaml_p['wind_info']:
            self.world_compressed *= 0

        #self.set_measurement() #already happened in update()
        if not yaml_p['measurement_info']:
            self.measurement *= 0

        total_z = (self.ceiling-(self.position_est[2]-self.height_above_ground(est=True)))/self.size_z
        boundaries = np.array([self.normalize_pos(self.position_est[0]-self.start[0]), self.normalize_pos(self.position_est[1]-self.start[1]), self.rel_pos_est, total_z])

        tar_x = int(np.clip(self.target[0],0,self.size_x - 1))
        tar_y = int(np.clip(self.target[1],0,self.size_y - 1))
        self.res_z_squished = (self.target[2]-self.world[0,tar_x,tar_y,0])/(self.ceiling - self.world[0,tar_x,tar_y,0]) - self.height_above_ground(est=True) / (self.dist_to_ceiling(est=True) + self.height_above_ground(est=True))

        if yaml_p['autoencoder'] != 'HAE_bidir':
            world_compressed = self.normalize_world(self.world_compressed)
        else:
            world_compressed = self.normalize_world(self.world_compressed)
            world_compressed[0:2] = self.world_compressed[0:2]*yaml_p['unit_xy']
            world_compressed[4:6] = self.world_compressed[4:6]*yaml_p['unit_xy']
        self.state = np.concatenate((self.normalize_pos(self.residual_est[0:2]),[self.res_z_squished], self.normalize_world(self.velocity_est).flatten(), boundaries.flatten(), self.normalize_world(self.measurement).flatten(), world_compressed), axis=0)

        self.bottleneck = len(self.state)
        self.state = self.state.astype(np.float32)

        if self.importance is not None:
            self.state[self.importance] = np.random.uniform(-1,1)

    def move_particle(self):
        self.U = 0
        not_done = True

        for n in range(self.n):
            lag_int = int(self.delay/yaml_p['delta_t'])
            lag_rest = self.delay - lag_int*yaml_p['delta_t']
            if lag_rest < n*yaml_p['delta_t_physics']:
                lag_int -= 1
            lag_int = np.clip(lag_int + 2,0,len(self.action_hist)) #extensively tested, I know it looks complicated, but it works
            action = self.action_hist[-lag_int]

            dist_bottom = self.height_above_ground()
            dist_top = self.dist_to_ceiling()
            self.rel_pos = dist_bottom / (dist_top + dist_bottom)
            if yaml_p['balloon'] == 'indoor_balloon':
                u = self.ll_controler.pid(action, self.rel_pos, self.p_z)
            elif yaml_p['balloon'] == 'outdoor_balloon':
                u = self.ll_controler.bangbang(action, self.rel_pos)

            #update physics model
            self.adapt_volume(u)

            b = self.net_force(u)/yaml_p['unit_z']/self.mass_total
            self.U += abs(u)*self.delta_tn
            coord = [int(i) for i in np.floor(self.position)]

            # calculate velocity at time step t
            w_x, w_y, w_z = self.interpolate(self.world_squished)

            # add noise
            avg_mag = np.mean(abs(self.world[1:4]))

            if yaml_p['environment'] == 'python3':
                noise = self.interpolate(self.noise,noise=True)
                n_x, n_y, n_z = avg_mag*self.prop_mag*noise*[1,1,0.25] #usually noise in z is a lot less than in x or y

                if yaml_p['3d'] == False: #because there is so much noise in x direction when it's a 2d field
                    n_x *= 1.3

            w_x += n_x
            w_y += n_y
            w_z += n_z

            # calculate new velocity
            v_x = (np.sign(w_x - self.p_x) * (w_x - self.p_x)**2 * self.c + 0)*self.delta_tn + self.p_x
            v_y = (np.sign(w_y - self.p_y) * (w_y - self.p_y)**2 * self.c + 0)*self.delta_tn + self.p_y
            v_z = (np.sign(w_z - self.p_z) * (w_z - self.p_z)**2 * self.c + b)*self.delta_tn + self.p_z

            # make sure that for big delta_tn this doesn't explode
            v_x = np.clip(v_x, -w_x, w_x)
            v_y = np.clip(v_y, -w_y, w_y)
            v_z = np.clip(v_z, -w_z + self.term_velocity[1], w_z + self.term_velocity[0])

            # update
            self.position += [v_x*self.delta_tn, v_y*self.delta_tn, v_z*self.delta_tn]
            self.p_x = v_x
            self.p_y = v_y
            self.p_z = v_z

            # write down path in history
            self.path.append(self.position.copy()) #because without copy otherwise it somehow overwrites it
            self.path_est.append(self.position_est.copy()) #because without copy otherwise it somehow overwrites it

            # find min_proj_dist
            self.residual = self.target - self.position

            if yaml_p['3d']:
                min_proj_dist = np.sqrt((self.residual[0]*self.render_ratio/self.radius_xy)**2 + (self.residual[1]*self.render_ratio/self.radius_xy)**2 + (self.residual[2]/self.radius_z)**2)
                min_dist = np.sqrt((self.residual[0]*self.render_ratio)**2 + (self.residual[1]*self.render_ratio)**2 + (self.residual[2])**2)*yaml_p['unit_z'] #in meters
            else:
                min_proj_dist = np.sqrt((self.residual[1]*self.render_ratio/self.radius_xy)**2 + (self.residual[2]/self.radius_z)**2)
                min_dist = np.sqrt((self.residual[1]*self.render_ratio)**2 + (self.residual[2])**2)*yaml_p['unit_z'] #in meters

            if min_proj_dist < self.min_proj_dist:
                self.min_proj_dist = min_proj_dist
                self.min_dist = min_dist

            # update time
            self.t -= yaml_p['delta_t_logger']/self.n

            # check if done or not
            if (self.position[0] < 0) | (self.position[0] > self.size_x - 1):
                not_done = False
            if (self.position[1] < 0) | (self.position[1] > self.size_y - 1):
                not_done = False
            if self.height_above_ground() < 0: #check if crashed into terrain
                not_done = False
            if self.dist_to_ceiling() < 0: #check if crashed into ceiling
                not_done = False
            if self.t < 0: #check if flight time is over
                not_done = False
            if self.battery_level < 0: #check if battery is empty
                not_done = False

            # update EKF
            force_est = (max(0,u)*self.delta_f_up + min(0,u)*self.delta_f_down)/yaml_p['unit_z']/self.mass_total
            self.update_est(force_est,self.c, self.delta_tn)
            self.set_measurement(self.est_x.wind(),self.est_y.wind())

        self.velocity = (self.position - self.path[-self.n])/yaml_p['delta_t']
        self.velocity_est = (self.position_est - self.path_est[-self.n])/yaml_p['delta_t']

        return not_done

    def live_particle(self):
        self.U = 0
        not_done = True

        data = {
        'action': self.action,
        'target': self.target.tolist(),
        'c': self.c,
        'ceiling': self.ceiling,
        'delta_f_up': self.delta_f_up,
        'delta_f_down': self.delta_f_down,
        'mass_total': self.mass_total
        }
        self.send(data) #write action to file

        #timing of the when to receive the data from the hardware
        delta_t = time.time() - self.real_time
        if yaml_p['delta_t_logger'] - delta_t > 0:
            time.sleep(yaml_p['delta_t_logger'] - delta_t)
        else:
            print('ERROR: Choose higher delta_t_logger')
        self.real_time = time.time()
        data = self.receive()

        # update
        self.position = np.array(data['position'])
        self.position_est = np.array(data['position_est'])
        [self.p_x, self.p_y, self.p_z] = data['velocity']

        # write down path in history
        self.path = np.array(data['path'])
        self.path_est = np.array(data['path_est'])

        self.min_proj_dist = data['min_proj_dist']
        self.min_dist = data['min_dist']

        # update time
        self.t -= yaml_p['delta_t_logger']

        # update EKF
        self.set_measurement(data['measurement'][0],data['measurement'][1])

        self.velocity = (self.position - self.position_old)/yaml_p['delta_t']
        self.velocity_est = (self.position_est - self.position_est_old)/yaml_p['delta_t']

        self.position_old = self.position
        self.position_est_old = self.position_est

        return not_done

    def adapt_volume(self,u):
        # pressure
        pressure_init = 101300 #Pa
        slope_pressure = -0.00010393333
        pressure = pressure_init + self.position[2]*yaml_p['unit_z']*slope_pressure #Pa

        # temperature
        temp_init = 15.00 #Pa
        slope_temp = -0.00064966666
        temp = temp_init + self.position[2]*yaml_p['unit_z']*slope_temp #Pa

        # viscosity
        vis_init = 1.87 #Pa s
        slope_vis = 0.0045
        vis = vis_init + temp*slope_vis #Pa s

        # density
        rho_air_init = 1.225 #kg/m^3
        rho_gas_init = 0.1785 #kg/m^3
        self.rho_air = rho_air_init*temp_init/temp*pressure_init/pressure
        self.rho_gas = rho_gas_init*temp_init/temp*pressure_init/pressure

        self.battery_level -= (self.rest_consumption*self.delta_tn + abs(min(u,0))*self.consumption_down*self.delta_tn + max(u,0)*self.consumption_up*self.delta_tn)/self.battery_capacity

        # volume
        volume_init = self.mass_structure/(rho_air_init - rho_gas_init) #m^3
        self.volume = volume_init*pressure_init/pressure*temp/temp_init #m^3
        self.diameter = 2*(self.volume*3/(4*np.pi))**(1/3) #m
        self.area = (self.diameter/2)**2*np.pi #m^2
        self.mass_total = self.mass_structure + volume_init*rho_gas_init #kg
        self.c = self.area*self.rho_air*self.c_w/(2*self.mass_total)/(1/yaml_p['unit_z'])

        # for update_particle
        self.term_velocity = np.array([np.sqrt(2*self.delta_f_up/(self.c_w*self.rho_air*self.area)), -np.sqrt(2*self.delta_f_down/(self.c_w*self.rho_air*self.area))])/yaml_p['unit_z']

    def net_force(self,u):
        f_balloon = (self.volume*(self.rho_air-self.rho_gas) - self.mass_structure)*9.81
        if u > 0:
            f_net = f_balloon + self.delta_f_up*u
        else:
            f_net = f_balloon + self.delta_f_down*u
        return f_net #N

    def height_above_ground(self, est=False):
        self.terrain_est = self.f_terrain(self.position_est[0], self.position_est[1])[0]
        self.terrain = self.f_terrain(self.position_est[0], self.position[1])[0] #is used by the logger
        if est:
            return self.position_est[2] - self.terrain_est
        else:
            return self.position[2] - self.terrain

    def set_ceiling(self):
        np.random.seed(self.seed) #this is needed so the same ceiling is used when the target is set
        self.seed +=1
        self.ceiling = np.random.uniform(1-yaml_p['ceiling_width'], 1) * self.size_z
        np.random.seed() #this is needed so the rest of the code is still random

    def dist_to_ceiling(self, est=False):
        if est:
            return self.ceiling - self.position_est[2]
        else:
            return self.ceiling - self.position[2]

    def set_measurement(self,est_x_wind,est_y_wind):
        self.measurement = np.array([est_x_wind, est_y_wind])
        if np.linalg.norm(self.measurement) != 0:
            self.esterror_wind = np.linalg.norm(self.interpolate(self.world_squished)[0:2] - self.measurement) / np.linalg.norm(self.measurement)
        else:
            self.esterror_wind = np.inf

        self.rel_pos_est = self.height_above_ground(est=True)/(self.ceiling-(self.position_est[2]-self.height_above_ground(est=True)))
        if yaml_p['world_est']:
            self.set_world_est(self.rel_pos_est,self.measurement)

    def interpolate(self, world, position=None, noise=False):
        if position is None: #for self.proj_action()
            pos_z_squished = self.height_above_ground() / (self.dist_to_ceiling() + self.height_above_ground())*len(world[0,0,0,:])
            position = copy.copy(self.position)
            position[2] = pos_z_squished

            if noise:
                render_ratio = np.divide([yaml_p['unit_xy'], yaml_p['unit_xy'], yaml_p['unit_z']],[yaml_p['unit_noise_xy'], yaml_p['unit_noise_xy'], yaml_p['unit_noise_z']])
                position *= render_ratio

        coord_x = int(np.clip(position[0],0,len(world[0,:,0,0]) - 1))
        coord_y = int(np.clip(position[1],0,len(world[0,0,:,0]) - 1))
        coord_z = int(np.clip(position[2],0,len(world[0,0,0,:]) - 1))

        x = np.clip(position[0] - coord_x,0,1)
        y = np.clip(position[1] - coord_y,0,1)
        z = np.clip(position[2] - coord_z,0,1)

        # I detect runnning out of bounds in a later stage
        i_x = 1
        i_y = 1
        i_z = 1

        if coord_x == len(world[0,:,0,0])-1:
            i_x = 0
        if coord_y == len(world[0,0,:,0])-1:
            i_y = 0
        if coord_z == len(world[0,0,0,:])-1:
            i_z = 0

        f_000 = world[-4::,coord_x,coord_y,coord_z]
        f_001 = world[-4::,coord_x,coord_y,coord_z+i_z]
        f_010 = world[-4::,coord_x,coord_y+i_y,coord_z]
        f_011 = world[-4::,coord_x,coord_y+i_y,coord_z+i_z]
        f_100 = world[-4::,coord_x+i_x,coord_y,coord_z]
        f_101 = world[-4::,coord_x+i_x,coord_y,coord_z+i_z]
        f_110 = world[-4::,coord_x+i_x,coord_y+i_y,coord_z]
        f_111 = world[-4::,coord_x+i_x,coord_y+i_y,coord_z+i_z]

        interp = f_000*(1-x)*(1-y)*(1-z) + f_001*(1-x)*(1-y)*z + f_010*(1-x)*y*(1-z) + f_011*(1-x)*y*z + f_100*x*(1-y)*(1-z) + f_101*x*(1-y)*z + f_110*x*y*(1-z) + f_111*x*y*z

        w_x, w_y, w_z = interp[0:3] #don't care about the sigma from meteo swiss
        w_x /= yaml_p['unit_xy']
        w_y /= yaml_p['unit_xy']
        w_z /= yaml_p['unit_z']

        return np.array([w_x, w_y, w_z])

    def set_noise(self):
        if self.train_or_test == 'test':
            np.random.seed(self.seed)
            self.seed +=1
        path = yaml_p['noise_path'] + self.train_or_test + '/tensor'
        noise_name = np.random.choice(os.listdir(path))
        self.noise = torch.load(path + '/' + noise_name)

        size_n_x = len(self.noise[0])
        size_n_y = len(self.noise[0][0])
        size_n_z = len(self.noise[0][0][0])

        self.prop_mag = np.random.uniform(yaml_p['prop_mag_min'], yaml_p['prop_mag_max'])

    def update_est(self,u,c,delta_t):
        std = 0 #sensor noise
        if self.train_or_test == 'test':
            np.random.seed(self.seed)
            self.seed +=1
        noise = np.random.normal(0,std,3)

        self.est_x.one_cycle(0,self.position[0] + noise[0], c, delta_t)
        self.est_y.one_cycle(0,self.position[1] + noise[1], c, delta_t)
        self.est_z.one_cycle(u,self.position[2] + noise[2], c, delta_t)
        self.position_est = np.array([self.est_x.xhat_0[0], self.est_y.xhat_0[0], self.est_z.xhat_0[0]])

        if np.linalg.norm(self.position) != 0:
            self.esterror_pos = np.linalg.norm(self.position - self.position_est)/np.linalg.norm(self.position)
        else:
            self.esterror_pos = np.inf
        if np.linalg.norm(self.velocity) != 0:
            self.esterror_vel = np.linalg.norm(self.velocity - self.velocity_est)/np.linalg.norm(self.velocity)
        else:
            self.esterror_vel = np.inf

    def set_world_est(self, pos_z_squished, data):
        if self.t < self.T-yaml_p['delta_t']/5: #the first few measurements are rubbish
            idx = np.clip(int(pos_z_squished*self.world_est_bn),0,self.world_est_bn - 1)

            if idx > 0:
                check_min = abs(np.subtract(idx, self.world_est_mask[0:idx]))
            else:
                check_min = [0]
            if idx < self.world_est_bn - 1:
                check_max = abs(np.subtract(idx, self.world_est_mask[idx+1::]))
            else:
                check_max = [0]

            idx_low = np.clip(np.argmin(check_min),0,self.world_est_bn - 2)
            idx_high = np.clip(idx + 1 + np.argmin(check_max),1,self.world_est_bn - 1)

            flag_low = np.clip(self.world_est_mask[idx_low],0,1)
            flag_high = np.clip(self.world_est_mask[idx_high],0,1)

            if flag_low:
                range_low = int(np.ceil(idx - idx_low)/2)
            else:
                range_low = int(np.ceil(idx - idx_low))
            if flag_high:
                range_high = int(np.ceil(idx_high - idx)/2)
            else:
                range_high = int(np.ceil(idx_high - idx))

            #fill in data point
            start = np.clip(idx-range_low,0,idx)
            stop = np.clip(idx+1+range_high,idx,self.world_est_bn - 1)
            self.world_est_data[:,start:stop] = np.array([data]*(stop - start)).T*yaml_p['unit_xy']
            self.world_est_mask[idx] = idx

    def update_world_est(self):
        for i in range(self.world_est_bn):
            start = int(i*self.size_z/self.world_est_bn)
            stop = int((i + 1)*self.size_z/self.world_est_bn)
            step = stop - start
            self.world_est[1,:,:,start:stop] = np.resize(self.world_est_data[0,i],(1,self.size_x, self.size_y, step))
            self.world_est[2,:,:,start:stop] = np.resize(self.world_est_data[1,i],(1,self.size_x, self.size_y, step))
        self.world_est[1,:,:,start::] = np.resize(self.world_est_data[0,-1],(1,self.size_x, self.size_y, step))
        self.world_est[2,:,:,start::] = np.resize(self.world_est_data[1,-1],(1,self.size_x, self.size_y, step))

        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        cmap = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)

        coord_x = int(np.clip(self.position[0],0,self.size_x - 1))
        coord_y = int(np.clip(self.position[1],0,self.size_y - 1))
        coord_z = int(np.clip(self.position[2],0,self.size_z - 1))

        fig, ax = plt.subplots(2)
        mag = 1
        ax[1].imshow(self.world_est[1,:,coord_y,:].T, origin='lower', cmap=cmap, alpha=0.7, vmin=-mag, vmax=mag)
        ax[1].set_aspect(yaml_p['unit_z']/yaml_p['unit_xy'])
        ax[0].imshow(self.world_est[2,coord_x,:,:].T, origin='lower', cmap=cmap, alpha=0.7, vmin=-mag, vmax=mag)
        ax[0].set_aspect(yaml_p['unit_z']/yaml_p['unit_xy'])

        plt.savefig('debug_imshow.png')
        plt.close()
        """

    def proj_action(self, position, target):
        res = self.size_z
        proj = []
        for i in range(res):
            pos_z = i/res*self.size_z

            if yaml_p['3d']:
                wind = self.interpolate(self.world_squished,position=[position[0], position[1], pos_z])[0:2]
                residual = (target - position)[0:2]
            else:
                wind = self.interpolate(self.world_squished,position=[position[0], position[1], pos_z])[1:2]
                residual = (target - position)[1:2]
            if np.linalg.norm(residual) != 0:
                proj.append(np.dot(wind, residual)/np.linalg.norm(residual)**2)
            else:
                proj.append(1)
        return proj

    def normalize_world(self, x):
        x = np.array(x)
        if yaml_p['balloon'] == 'outdoor_balloon':
            c = 0.005
        elif yaml_p['balloon'] == 'indoor_balloon':
            c = 0.5
        else:
            print('ERROR: Choose an existing balloon type')
        return x/(abs(x) + c)

    def normalize_pos(self, x):
        x = np.array(x)
        c = np.max([self.size_x,self.size_y])/4
        return x/(abs(x) + c)

    def send(self, data):
        path = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/communication/'
        with open(path + 'action.txt', 'w') as f:
            f.write(json.dumps(data))
        return data

    def receive(self):
        successful = False
        path = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/communication/'
        while not os.path.isfile(path + 'data.txt'):
            print('waiting for the hardware to publish')
            time.sleep(1)
        start = time.time()
        corrupt = False
        while not successful:
            with open(path + 'data.txt') as json_file:
                try:
                    data = json.load(json_file)
                    successful = True
                except:
                    corrupt = True
        if corrupt:
            print('data corrupted, lag of ' + str(np.round(time.time() - start,3)) + '[s]')
        return data
