import numpy as np
import scipy
import copy
import torch
import os

from preprocess_wind import squish
from build_ll_controller import ll_controler

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

class character():
    def __init__(self, size_x, size_y, size_z, start, target, radius_xy, radius_z, T, world, world_compressed, train_or_test, seed):
        self.render_ratio = yaml_p['unit_xy'] / yaml_p['unit_z']
        self.radius_xy = radius_xy
        self.radius_z = radius_z

        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.start = start.astype(float)
        self.target = target.astype(float)

        self.ll_controler = ll_controler()

        if yaml_p['balloon'] == 'outdoor_balloon':
            self.mass_structure = 1 #kg
            self.delta_f = yaml_p['delta_f'] #N
            self.ascent_consumption = 15 #W
            self.descent_consumption = 15 #W
            self.rest_consumption = 0.5 #W
            self.battery_capacity = 263736 #Ws #100000

        elif yaml_p['balloon'] == 'indoor_balloon':
            self.mass_structure = 1.2 #kg
            self.delta_f = 0.01 #N
            self.ascent_consumption = 5 #W
            self.descent_consumption = 2.5 #W
            self.rest_consumption = 0.5 #W
            self.battery_capacity = 1798 #Ws
        else:
            print('ERROR: please choose one of the available balloons')

        self.t = T
        self.battery_level = 1
        self.action = 1
        self.diameter = 0

        self.world = world
        self.world_compressed = world_compressed

        self.train_or_test = train_or_test

        self.position = copy.copy(self.start)
        self.velocity = np.array([0,0,0])

        self.n = int(yaml_p['delta_t']*1/0.5) #physics every 1/x seconds
        self.delta_tn = yaml_p['delta_t']/self.n
        self.m = 1 #running mean step size

        # interpolation for terrain
        x = np.linspace(0,self.size_x,len(self.world[0,:,0,0]))
        y = np.linspace(0,self.size_y,len(self.world[0,0,:,0]))

        self.f_terrain = scipy.interpolate.interp2d(x,y,self.world[0,:,:,0].T)

        self.seed = seed
        self.set_ceiling()
        self.world_squished = squish(self.world, self.ceiling)

        if yaml_p['W_20'] != 0:
            self.set_noise()

        self.residual = self.target - self.position
        self.measurement = np.array([0,0])

        self.importance = None
        self.set_state()

        self.path = [self.position.copy(), self.position.copy()]
        self.velocity_hist = [self.velocity.copy()]

        self.min_proj_dist = np.inf
        self.min_proj_dist = np.sqrt((self.residual[0]*self.render_ratio/self.radius_xy)**2 + (self.residual[1]*self.render_ratio/self.radius_xy)**2 + (self.residual[2]/self.radius_z)**2)

        #for move_particle (previous velocity is zero at the beginning)
        self.p_x = 0
        self.p_y = 0
        self.p_z = 0

    def update(self, action, world, world_compressed):
        self.action = action
        self.world = world
        self.world_compressed = world_compressed

        not_done = self.move_particle()

        # update state
        self.residual = self.target - self.position
        self.set_measurement()

        self.set_state()

        return not_done

    def set_state(self):
        if not yaml_p['wind_info']:
            self.world_compressed *= 0

        if not yaml_p['measurement_info']:
            self.measurement *= 0

        rel_pos = self.height_above_ground()/(self.ceiling-(self.position[2]-self.height_above_ground()))
        total_z = (self.ceiling-(self.position[2]-self.height_above_ground()))/self.size_z

        if yaml_p['position_info']:
            boundaries = np.array([self.normalize_map(self.position[0]-self.start[0]), self.normalize_map(self.position[1]-self.start[1]), rel_pos, total_z])
        else:
            boundaries = np.array([rel_pos, total_z])

        tar_x = int(np.clip(self.target[0],0,self.size_x - 1))
        tar_y = int(np.clip(self.target[1],0,self.size_y - 1))
        self.res_z_squished = (self.target[2]-self.world[0,tar_x,tar_y,0])/(self.ceiling - self.world[0,tar_x,tar_y,0]) - self.height_above_ground() / (self.dist_to_ceiling() + self.height_above_ground())

        self.state = np.concatenate((self.normalize_map(self.residual[0:2]),[self.res_z_squished], self.normalize(self.velocity).flatten(), boundaries.flatten(), self.normalize(self.measurement).flatten(), self.normalize(self.world_compressed).flatten()), axis=0)

        self.bottleneck = len(self.state)
        self.state = self.state.astype(np.float32)

        if self.importance is not None:
            self.state[self.importance] = np.random.uniform(-1,1)

    def move_particle(self):
        self.U = 0
        not_done = True
        for n in range(self.n):
            dist_bottom = self.height_above_ground()
            dist_top = self.dist_to_ceiling()
            rel_pos = dist_bottom / (dist_top + dist_bottom)
            u = self.ll_controler.bangbang(self.action,rel_pos)

            #update physics model
            self.adapt_volume(u)

            c = self.area*self.rho_air*self.c_w/(2*self.mass_total)

            b = self.net_force(u)/yaml_p['unit_z']**2/self.mass_total
            self.U += abs(u)/self.n

            coord = [int(i) for i in np.floor(self.position)]

            # calculate velocity at time step t
            w_x, w_y, w_z, sig_xz = self.interpolate_wind()

            """
            x = np.arange(0,self.size_x,1)
            y = np.arange(0,self.size_y,1)
            z = np.arange(0,self.size_z,1)
            """

            if yaml_p['W_20'] != 0:
                w_x, w_y, w_z = self.add_noise(w_x, w_y, w_z)

            v_x = (np.sign(w_x - self.p_x) * (w_x - self.p_x)**2 * c + 0)*self.delta_tn + self.p_x
            v_y = (np.sign(w_y - self.p_y) * (w_y - self.p_y)**2 * c + 0)*self.delta_tn + self.p_y
            v_z = (np.sign(w_z - self.p_z) * (w_z - self.p_z)**2 * c + b)*self.delta_tn + self.p_z

            # update
            self.position += [v_x*self.delta_tn, v_y*self.delta_tn, v_z*self.delta_tn]
            self.p_x = v_x
            self.p_y = v_y
            self.p_z = v_z

            # write down path in history
            self.path.append(self.position.copy()) #because without copy otherwise it somehow overwrites it

            # find min_proj_dist
            self.residual = self.target - self.position
            min_proj_dist = np.sqrt((self.residual[0]*self.render_ratio/self.radius_xy)**2 + (self.residual[1]*self.render_ratio/self.radius_xy)**2 + (self.residual[2]/self.radius_z)**2)
            if min_proj_dist < self.min_proj_dist:
                self.min_proj_dist = min_proj_dist

            # update time
            self.t -= yaml_p['delta_t']/self.n

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

            if n%self.m == 0:
                self.velocity = (self.path[-1] - self.path[-self.m-1])/(self.m*self.delta_tn)
                # set velocity for state
                self.velocity_hist.append(self.velocity)

        return not_done

    def adapt_volume(self,u):
        #general properties
        self.c_w = 0.45

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

        self.battery_level -= (self.rest_consumption*self.delta_tn + abs(min(u,0))*self.descent_consumption*self.delta_tn + max(u,0)*self.ascent_consumption*self.delta_tn)/self.battery_capacity

        # volume
        volume_init = self.mass_structure/(rho_air_init - rho_gas_init) #m^3
        self.volume = pressure_init*volume_init/temp_init*temp/pressure #m^3
        self.diameter = 2*(self.volume*3/(4*np.pi))**(1/3) #m
        self.area = (self.diameter/2)**2*np.pi #m^2
        self.mass_total = self.mass_structure + volume_init*rho_gas_init #kg

    def net_force(self,u):
        f_balloon = (self.volume*(self.rho_air-self.rho_gas) - self.mass_structure)*9.81
        f_net = f_balloon + self.delta_f*u
        return f_net

    def height_above_ground(self):
        return self.position[2] - self.f_terrain(self.position[0], self.position[1])[0]

    def set_ceiling(self):
        if self.train_or_test == 'test':
            np.random.seed(self.seed)
            self.seed +=1
        self.ceiling = np.random.uniform(0.9, 1) * self.size_z

    def dist_to_ceiling(self):
        return self.ceiling - self.position[2]

    def set_measurement(self):
        v_t = self.velocity_hist[-1]
        v_prev = self.velocity_hist[-2]
        v_w = np.sign(v_t - v_prev)*((abs(v_t - v_prev))*self.mass_total/(self.m*self.delta_tn)*2/(self.c_w*self.area*self.rho_air))**(1/2) + (v_t + v_prev)/2
        self.measurement = v_w[0:2]

    def interpolate_wind(self):
        world = self.world_squished

        pos_z_squished = self.height_above_ground() / (self.dist_to_ceiling() + self.height_above_ground())*len(world[0,0,0,:])
        coord_x = int(np.clip(self.position[0],0,self.size_x - 1))
        coord_y = int(np.clip(self.position[1],0,self.size_y - 1))
        coord_z = int(np.clip(pos_z_squished,0,len(world[0,0,0,:])-1))

        x = self.position[0] - coord_x
        y = self.position[1] - coord_y
        z = pos_z_squished - coord_z

        # I detect runnning out of bounds in a later stage
        if coord_x == self.size_x-1:
            coord_x -= 1
            x = 1
        if coord_y == self.size_y-1:
            coord_y -= 1
            y = 1
        if coord_z == self.size_z-1:
            coord_z -= 1
            z = 1

        f_000 = world[-4::,coord_x,coord_y,coord_z]
        f_001 = world[-4::,coord_x,coord_y,coord_z+1]
        f_010 = world[-4::,coord_x,coord_y+1,coord_z]
        f_011 = world[-4::,coord_x,coord_y+1,coord_z+1]
        f_100 = world[-4::,coord_x+1,coord_y,coord_z]
        f_101 = world[-4::,coord_x+1,coord_y,coord_z+1]
        f_110 = world[-4::,coord_x+1,coord_y+1,coord_z]
        f_111 = world[-4::,coord_x+1,coord_y+1,coord_z+1]

        wind = f_000*(1-x)*(1-y)*(1-z) + f_001*(1-x)*(1-y)*z + f_010*(1-x)*y*(1-z) + f_011*(1-x)*y*z + f_100*x*(1-y)*(1-z) + f_101*x*(1-y)*z + f_110*x*y*(1-z) + f_111*x*y*z

        w_x, w_y, w_z, sig_xz = wind
        w_x /= yaml_p['unit_xy']
        w_y /= yaml_p['unit_xy']
        w_z /= yaml_p['unit_z']

        return np.array([w_x, w_y, w_z, sig_xz])

    def set_noise(self):
        if self.train_or_test == 'test':
            np.random.seed(self.seed)
            self.seed +=1
        path = yaml_p['noise_path'] + self.train_or_test + '/tensor_' + str(yaml_p['W_20'])
        noise_name = np.random.choice(os.listdir(path))
        self.noise = torch.load(path + '/' + noise_name)

    def add_noise(self, w_x, w_y, w_z):
        size_n_x = len(self.noise[0])
        size_n_y = len(self.noise[0][0])
        size_n_z = len(self.noise[0][0][0])

        if (round(size_n_x/self.render_ratio) != self.size_x) | (round(size_n_y/self.render_ratio) != self.size_y) | (round(size_n_z) != self.size_z):
            print("ERROR: size of noise map doesn't match the one of the world map")

        rel_pos = self.height_above_ground()/(self.ceiling-(self.position[2]-self.height_above_ground()))
        position_n = self.position[0:2]*[self.render_ratio, self.render_ratio]
        position_n = np.append(position_n, rel_pos*size_n_z)

        position_n[0] = np.clip(position_n[0],0,size_n_x - 1)
        position_n[1] = np.clip(position_n[1],0,size_n_y - 1)
        position_n[2] = np.clip(position_n[2],0,size_n_z - 1)

        coord_n = [int(i) for i in np.floor(position_n)]

        w_x += self.noise[0][coord_n[0], coord_n[1], coord_n[2]]/yaml_p['unit_xy']
        w_y += self.noise[1][coord_n[0], coord_n[1], coord_n[2]]/yaml_p['unit_xy']
        w_z += self.noise[2][coord_n[0], coord_n[1], coord_n[2]]/yaml_p['unit_z']

        return w_x, w_y, w_z

    def normalize(self,x):
        x = np.array(x)
        return x/(abs(x)+0.005)

    def normalize_map(self,x):
        x = np.array(x)
        return x/(abs(x)+5)
