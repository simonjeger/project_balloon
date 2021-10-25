import numpy as np
import scipy
import copy
import torch
import os
from human_autoencoder import HAE
from build_autoencoder import VAE
from scipy.interpolate import NearestNDInterpolator

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
        self.n = int(yaml_p['delta_t']/yaml_p['delta_t_physics'])
        self.delta_tn = yaml_p['delta_t']/self.n

        self.render_ratio = yaml_p['unit_xy'] / yaml_p['unit_z']
        self.radius_xy = radius_xy
        self.radius_z = radius_z

        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.start = start.astype(float)
        self.target = target.astype(float)

        self.ll_controler = ll_controler()
        self.est_x = ekf(self.delta_tn)
        self.est_y = ekf(self.delta_tn)
        self.est_z = ekf(self.delta_tn)

        self.esterror_pos = 0
        self.esterror_vel = 0
        self.esterror_wind = 0

        if yaml_p['balloon'] == 'outdoor_balloon':
            self.mass_structure = 0.6 #kg
            self.delta_f = 1 #N
            self.ascent_consumption = 15 #W
            self.descent_consumption = 15 #W
            self.rest_consumption = 0.5 #W
            self.battery_capacity = 263736 #Ws #100000

        elif yaml_p['balloon'] == 'indoor_balloon':
            self.mass_structure = 0.06 #kg
            self.delta_f = 0.01 #N
            self.ascent_consumption = 5 #W
            self.descent_consumption = 2.5 #W
            self.rest_consumption = 0.5 #W
            self.battery_capacity = 1798 #Ws
        else:
            print('ERROR: please choose one of the available balloons')

        # initialize autoencoder object
        if yaml_p['autoencoder'][0:3] == 'HAE':
            self.ae = HAE()
        if yaml_p['autoencoder'] == 'VAE':
            self.ae = VAE()
            self.ae.load_weights('autoencoder/model_' + str(yaml_p['vae_nr']) + '.pt')

        self.t = T
        self.battery_level = 1
        self.action = 1
        self.diameter = 0

        self.world = world

        self.world_est = np.zeros_like(self.world)
        self.world_est[0] = self.world[0] #terrain is known

        self.measurement_hist_u = []
        self.measurement_hist_v = []
        self.moment_hist = []
        self.w_est_xy = yaml_p['unit_xy']
        self.w_est_z = yaml_p['unit_z']*100
        self.w_est_t = yaml_p['delta_t']*0.00001

        self.train_or_test = train_or_test

        self.position = copy.copy(self.start)
        self.position_est = copy.copy(self.position)
        self.velocity = np.array([0,0,0])
        self.velocity_est = np.array([0,0,0])

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
        self.measurement = self.interpolate(self.world_squished)[0:2]

        self.importance = None
        self.set_state()

        self.path = [self.position.copy(), self.position.copy()]
        self.path_est = [self.position.copy(), self.position.copy()]

        self.min_proj_dist = np.inf
        self.min_proj_dist = np.sqrt((self.residual[0]*self.render_ratio/self.radius_xy)**2 + (self.residual[1]*self.render_ratio/self.radius_xy)**2 + (self.residual[2]/self.radius_z)**2)

        #for move_particle (previous velocity is zero at the beginning)
        self.p_x = 0
        self.p_y = 0
        self.p_z = 0

    def update(self, action, world):
        self.action = action
        self.world = world
        self.world_squished = squish(self.world, self.ceiling)

        not_done = self.move_particle()

        # update state
        self.set_state()
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

        rel_pos_est = self.height_above_ground(est=True)/(self.ceiling-(self.position_est[2]-self.height_above_ground(est=True)))
        total_z = (self.ceiling-(self.position_est[2]-self.height_above_ground(est=True)))/self.size_z
        boundaries = np.array([self.normalize_map(self.position_est[0]-self.start[0]), self.normalize_map(self.position_est[1]-self.start[1]), rel_pos_est, total_z])

        tar_x = int(np.clip(self.target[0],0,self.size_x - 1))
        tar_y = int(np.clip(self.target[1],0,self.size_y - 1))
        self.res_z_squished = (self.target[2]-self.world[0,tar_x,tar_y,0])/(self.ceiling - self.world[0,tar_x,tar_y,0]) - self.height_above_ground(est=True) / (self.dist_to_ceiling(est=True) + self.height_above_ground(est=True))

        if yaml_p['autoencoder'] != 'HAE_bidir':
            world_compressed = self.normalize(self.world_compressed)
        else:
            world_compressed = self.normalize(self.world_compressed)
            world_compressed[0:2] = self.world_compressed[0:2]*yaml_p['unit_xy']
            world_compressed[4:6] = self.world_compressed[4:6]*yaml_p['unit_xy']
        self.state = np.concatenate((self.normalize_map(self.residual_est[0:2]),[self.res_z_squished], self.normalize(self.velocity_est).flatten(), boundaries.flatten(), self.normalize(self.measurement).flatten(), world_compressed), axis=0)

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
            u = self.ll_controler.pid(self.action,rel_pos, self.p_z)

            #update physics model
            self.adapt_volume(u)

            c = self.area*self.rho_air*self.c_w/(2*self.mass_total)

            b = self.net_force(u)/yaml_p['unit_z']**2/self.mass_total
            self.U += abs(u)/self.n

            coord = [int(i) for i in np.floor(self.position)]

            # calculate velocity at time step t
            w_x, w_y, w_z = self.interpolate(self.world_squished)

            # add noise
            if yaml_p['W_20'] != 0:
                n_x, n_y, n_z = self.interpolate(self.noise)
                w_x += n_x
                w_y += n_y
                w_z += n_z

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
            self.path_est.append(self.position_est.copy()) #because without copy otherwise it somehow overwrites it

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

            # update EKF
            self.update_est(u,c)
            self.set_measurement()

        self.velocity = (self.position - self.path[-self.n])/yaml_p['delta_t']
        self.velocity_est = (self.position_est - self.path_est[-self.n])/yaml_p['delta_t']

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
        self.volume = volume_init*pressure_init/pressure*temp/temp_init #m^3
        self.diameter = 2*(self.volume*3/(4*np.pi))**(1/3) #m
        self.area = (self.diameter/2)**2*np.pi #m^2
        self.mass_total = self.mass_structure + volume_init*rho_gas_init #kg

    def net_force(self,u):
        f_balloon = (self.volume*(self.rho_air-self.rho_gas) - self.mass_structure)*9.81
        f_net = f_balloon + self.delta_f*u
        return f_net

    def height_above_ground(self, est=False):
        if est:
            return self.position_est[2] - self.f_terrain(self.position_est[0], self.position_est[1])[0]
        else:
            return self.position[2] - self.f_terrain(self.position[0], self.position[1])[0]

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

    def set_measurement(self):
        self.measurement = np.array([self.est_x.wind(), self.est_y.wind()])
        if np.linalg.norm(self.measurement) != 0:
            self.esterror_wind = np.linalg.norm(self.interpolate(self.world_squished)[0:2] - self.measurement) / np.linalg.norm(self.measurement)
        else:
            self.esterror_wind = np.inf

        self.measurement_hist_u.append(self.measurement[0])
        self.measurement_hist_v.append(self.measurement[1])
        rel_pos_est = self.height_above_ground(est=True)/(self.ceiling-(self.position_est[2]-self.height_above_ground(est=True)))
        self.moment_hist.append([self.position_est[0]*self.w_est_xy, self.position_est[1]*self.w_est_xy, rel_pos_est*self.size_z*self.w_est_z, self.t*self.w_est_t])

    def interpolate(self, world):
        pos_z_squished = self.height_above_ground() / (self.dist_to_ceiling() + self.height_above_ground())*len(world[0,0,0,:])
        coord_x = int(np.clip(self.position[0],0,self.size_x - 1))
        coord_y = int(np.clip(self.position[1],0,self.size_y - 1))
        coord_z = int(np.clip(pos_z_squished,0,len(world[0,0,0,:])-1))

        x = np.clip(self.position[0] - coord_x,0,1)
        y = np.clip(self.position[1] - coord_y,0,1)
        z = np.clip(self.position[2] - coord_z,0,1)

        # I detect runnning out of bounds in a later stage
        i_x = 1
        i_y = 1
        i_z = 1

        if coord_x == self.size_x-1:
            i_x = 0
        if coord_y == self.size_y-1:
            i_y = 0
        if coord_z == self.size_z-1:
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
        path = yaml_p['noise_path'] + self.train_or_test + '/tensor_' + str(yaml_p['W_20'])
        noise_name = np.random.choice(os.listdir(path))
        self.noise = torch.load(path + '/' + noise_name)

        size_n_x = len(self.noise[0])
        size_n_y = len(self.noise[0][0])
        size_n_z = len(self.noise[0][0][0])

        if (size_n_x != self.size_x) | (size_n_y != self.size_y) | (size_n_z != self.size_z):
            print("ERROR: size of noise map doesn't match the one of the world map")

    def update_est(self,u,c):
        std = 0 #sensor noise
        if self.train_or_test == 'test':
            np.random.seed(self.seed)
            self.seed +=1
        noise = np.random.normal(0,std,3)

        self.est_x.one_cycle(0,c,self.position[0] + noise[0])
        self.est_y.one_cycle(0,c,self.position[1] + noise[1])
        self.est_z.one_cycle(u,c,self.position[2] + noise[2])
        self.position_est = np.array([self.est_x.xhat_0[0], self.est_y.xhat_0[0], self.est_z.xhat_0[0]])

        if np.linalg.norm(self.position) != 0:
            self.esterror_pos = np.linalg.norm(self.position - self.position_est)/np.linalg.norm(self.position)
        else:
            self.esterror_pos = np.inf
        if np.linalg.norm(self.velocity) != 0:
            self.esterror_vel = np.linalg.norm(self.velocity - self.velocity_est)/np.linalg.norm(self.velocity)
        else:
            self.esterror_vel = np.inf

    def update_world_est(self):
        if len(self.moment_hist) > self.n: #the first n measurements are rubbish, as the balloon didn't move yet
            interp_u = NearestNDInterpolator(self.moment_hist[self.n::], self.measurement_hist_u[self.n::])
            interp_v = NearestNDInterpolator(self.moment_hist[self.n::], self.measurement_hist_v[self.n::])

            X = np.linspace(0, self.size_x*self.w_est_xy, self.size_x)
            Y = np.linspace(0, self.size_y*self.w_est_xy, self.size_y)
            Z = np.linspace(0, self.size_z*self.w_est_z, self.size_z)
            T = np.linspace(yaml_p['T']*self.w_est_t, self.t*self.w_est_t, max(int((yaml_p['T'] - self.t)/yaml_p['delta_t']),1))
            X, Y, Z, T = np.meshgrid(X, Y, Z, T)  # 3D grid for interpolation

            u = interp_u(X, Y, Z, T)[:,:,:,-1].reshape(self.size_x, self.size_y, self.size_z) #to drop the time dimension
            v = interp_v(X, Y, Z, T)[:,:,:,-1].reshape(self.size_x, self.size_y, self.size_z)

            self.world_est[1:3] = [u*yaml_p['unit_xy'],v*yaml_p['unit_xy']]

            """
            import matplotlib.pyplot as plt
            import seaborn as sns
            cmap = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)

            for i in range(self.size_x):
                fig, ax = plt.subplots(2)
                for j in range(len(self.moment_hist[self.n::])):
                    if i*self.w_est_xy <= int(self.moment_hist[j][0]) < (i+1)*self.w_est_xy:
                        ax[0].scatter(self.moment_hist[j][0]/self.w_est_xy,self.moment_hist[j][2]/self.w_est_z, s=0.1, c='white') #c=self.measurement_hist_u[j]
                ax[0].imshow(np.multiply(u[:,i,:].T,yaml_p['unit_xy']), origin='lower', cmap=cmap, alpha=0.7, vmin=-10, vmax=10)
                ax[0].set_aspect(yaml_p['unit_z']/yaml_p['unit_xy'])


                ax[1].plot(np.multiply(self.measurement_hist_u[self.n::],yaml_p['unit_xy']))
                plt.savefig('debug_imshow_' + str(i) + '.png')
                plt.close()
            """

    def normalize(self,x):
        x = np.array(x)
        return x/(abs(x)+0.005)

    def normalize_map(self,x):
        x = np.array(x)
        return x/(abs(x)+5)
