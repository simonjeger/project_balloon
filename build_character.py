import numpy as np
import scipy
import random
from random import gauss
import copy
from scipy.ndimage import gaussian_filter

from preprocess_wind import squish
from lowlevel_controller import ll_pd

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

class character():
    def __init__(self, size_x, size_y, size_z, start, target, radius_xy, radius_z, T, world, world_compressed):
        self.render_ratio = yaml_p['unit_xy'] / yaml_p['unit_z']
        self.radius_xy = radius_xy
        self.radius_z = radius_z

        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.start = start.astype(float)
        self.target = target.astype(float)

        if yaml_p['balloon'] == 'outdoor_balloon':
            self.mass_structure = 2.5 #kg
            self.delta_volume = 0.012 #m^3
            self.pump_volume = 0.000083 #m^3
            self.pump_consumption = 5 #W
            self.valve_consumption = 2.5 #W
            self.rest_consumption = 0.5 #W
            self.battery_capacity = 13187 #Ws
            self.pressure_tank = 200000 #Pa
            self.l_in = 0.1 #m
            self.r_in = 0.00635 #m
        elif yaml_p['balloon'] == 'indoor_balloon':
            self.mass_structure = 1.2 #kg
            self.delta_volume = 0.006 #m^3
            self.pump_volume = 0.000083 #m^3
            self.pump_consumption = 5 #W
            self.valve_consumption = 2.5 #W
            self.rest_consumption = 0.5 #W
            self.battery_capacity = 1798 #Ws
            self.pressure_tank = 200000 #Pa
            self.l_in = 0.1 #m
            self.r_in = 0.00635 #m

        self.t = T
        self.battery_level = 1
        self.delta_v_prev = 0
        self.action = 1
        self.diameter = 0

        self.world = world
        self.world_compressed = world_compressed

        self.position = copy.copy(self.start)
        self.velocity = np.array([0,0,0])

        self.n = int(yaml_p['delta_t']/6)
        self.delta_tn = yaml_p['delta_t']/self.n

        # interpolation for terrain
        x = np.linspace(0,self.size_x,len(self.world[0,:,0,0]))
        y = np.linspace(0,self.size_y,len(self.world[0,0,:,0]))

        self.f_terrain = scipy.interpolate.interp2d(x,y,self.world[0,:,:,0].T)

        self.set_ceiling()
        self.world_squished = squish(self.world, self.ceiling)

        self.residual = self.target - self.position
        self.measurement = self.interpolate_wind(measurement=True)[0:3]

        self.importance = None
        self.set_state()

        self.path = [self.position.copy(), self.position.copy()]
        self.min_proj_dist = np.inf
        self.min_proj_dist = np.sqrt((self.residual[0]*self.render_ratio/self.radius_xy)**2 + (self.residual[1]*self.render_ratio/self.radius_xy)**2 + (self.residual[2]/self.radius_z)**2)

    def update(self, action, world_compressed, roll_out=False):
        self.action = action
        self.world_compressed = world_compressed

        not_done = self.move_particle(roll_out)
        if self.height_above_ground() < 0: #check if crashed into terrain
            not_done = False
        if self.dist_to_ceiling() < 0: #check if crashed into ceiling
            not_done = False
        if self.battery_level < 0: #check if battery is empty
            not_done = False

        # update state
        self.residual = self.target - self.position
        self.measurement = self.interpolate_wind(measurement=True)[0:3]

        self.set_state()

        # reduce flight length by 1 second
        self.t -= 1

        return not_done

    def set_state(self):
        if not yaml_p['wind_info']:
            self.world_compressed *= 0

        if not yaml_p['measurement_info']:
            self.measurement *= 0

        rel_pos = self.height_above_ground()/(self.ceiling-(self.position[2]-self.height_above_ground()))
        total_z = (self.ceiling-(self.position[2]-self.height_above_ground()))/self.size_z
        boundaries = np.array([rel_pos, total_z, self.position[0] - np.floor(self.position[0]), self.position[1] - np.floor(self.position[1])])

        tar_x = int(np.clip(self.target[0],0,self.size_x - 1))
        tar_y = int(np.clip(self.target[1],0,self.size_y - 1))
        self.res_z_squished = (self.target[2]-self.world[0,tar_x,tar_y,0])/(self.ceiling - self.world[0,tar_x,tar_y,0]) - self.height_above_ground() / (self.dist_to_ceiling() + self.height_above_ground())

        self.state = np.concatenate(((self.residual[0:2]/[self.size_x,self.size_y]).flatten(),[self.res_z_squished], self.normalize(self.velocity).flatten(), boundaries.flatten(), self.normalize(self.measurement).flatten(), self.normalize(self.world_compressed).flatten()), axis=0)

        self.bottleneck = len(self.state)
        self.state = self.state.astype(np.float32)

        if self.importance is not None:
            self.state[self.importance] = np.random.uniform(-1,1)

    def move_particle(self, roll_out):
        self.U = 0
        for _ in range(self.n):
            dist_bottom = self.height_above_ground()
            dist_top = self.dist_to_ceiling()
            rel_pos = dist_bottom / (dist_top + dist_bottom)
            velocity = self.velocity[2]
            u = ll_pd(self.action,rel_pos,velocity)

            #update physics model
            self.adapt_volume(u)

            c = self.area*self.rho_air*self.c_w/(2*self.mass_total)
            p_x = (self.path[-1][0] - self.path[-2][0])/self.delta_tn
            p_y = (self.path[-1][1] - self.path[-2][1])/self.delta_tn
            p_z = (self.path[-1][2] - self.path[-2][2])/self.delta_tn

            b = self.volume_to_force(self.delta_v)/yaml_p['unit_z']/self.mass_total
            self.U += abs(u)/self.n

            coord = [int(i) for i in np.floor(self.position)]
            not_done = (0 <= self.position[0] < self.size_x) & (0 <= self.position[1] < self.size_y) & (0 <= self.position[2] < self.size_z) #if still within bounds
            if not_done:
                # calculate velocity at time step t
                w_x, w_y, w_z, sig_xz = self.interpolate_wind()

                x = np.arange(0,self.size_x,1)
                y = np.arange(0,self.size_y,1)
                z = np.arange(0,self.size_z,1)

                w_x += gauss(0,sig_xz/np.sqrt(self.n)) #is it /sqrt(n) or just /n?
                w_y += gauss(0,sig_xz/np.sqrt(self.n)) #is it /sqrt(n) or just /n?
                w_z += gauss(0,sig_xz/np.sqrt(self.n))

                v_x = (np.sign(w_x - p_x) * (w_x - p_x)**2 * c + 0)*self.delta_tn + p_x
                v_y = (np.sign(w_y - p_y) * (w_y - p_y)**2 * c + 0)*self.delta_tn + p_y
                v_z = (np.sign(w_z - p_z) * (w_z - p_z)**2 * c + b)*self.delta_tn + p_z

                # update
                self.position += [v_x*self.delta_tn, v_y*self.delta_tn, v_z*self.delta_tn]
                p_x = v_x
                p_y = v_y
                p_z = v_z

                # set velocity for state
                self.velocity = np.array([v_x, v_y, v_z])

                # write down path in history
                self.path.append(self.position.copy()) #because without copy otherwise it somehow overwrites it

                # find min_proj_dist
                self.residual = self.target - self.position
                min_proj_dist = np.sqrt((self.residual[0]*self.render_ratio/self.radius_xy)**2 + (self.residual[1]*self.render_ratio/self.radius_xy)**2 + (self.residual[2]/self.radius_z)**2)
                if min_proj_dist < self.min_proj_dist:
                    self.min_proj_dist = min_proj_dist
        return not_done

    def adapt_volume(self,u):
        #general properties
        self.c_w = 0.45

        # density
        rho_air_init = 1.225 #kg/m^3
        rho_gas_init = 0.1785 #kg/m^3
        slope_gas = -0.00001052333 #linear interpolation between 0 and 30000m for air, assumption: it's the same for the lifting gas (https://www.engineeringtoolbox.com/standard-atmosphere-d_604.html)
        self.rho_air = rho_air_init + self.position[2]*yaml_p['unit_z']*slope_gas #kg/m^3
        self.rho_gas = rho_gas_init + self.position[2]*yaml_p['unit_z']*slope_gas #kg/m^3

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

        # gas flow
        gas_flow_in = (np.pi*self.r_in**4*(self.pressure_tank - pressure))/(8*vis*self.l_in) #m^3/s
        gas_flow_out = self.pump_volume #m^3/s

        self.delta_v = self.delta_v_prev + yaml_p['delta_t']*(max(u,0)*gas_flow_in + min(u,0)*gas_flow_out)
        self.delta_v = np.clip(self.delta_v,-self.delta_volume, self.delta_volume)
        self.battery_level -= (self.rest_consumption*self.delta_tn + abs(min(self.delta_v - self.delta_v_prev,0))*self.pump_consumption*self.delta_tn + max(self.delta_v - self.delta_v_prev,0)*self.valve_consumption*self.delta_tn)/self.battery_capacity
        self.delta_v_prev = self.delta_v

        volume = self.mass_structure/(self.rho_air - self.rho_gas) + self.delta_v #m^3
        self.diameter = 2*(volume*3/(4*np.pi))**(1/3) #m
        self.area = (self.diameter/2)**2*np.pi #m^2
        self.mass_total = self.mass_structure + volume*self.rho_gas #kg

        #print('Specs of ' + yaml_p['balloon'] + ': volume = ' + str(np.round(volume,2)) + 'm^3, self.diameter = ' + str(np.round(self.diameter,2)) + 'm, force = ' + str(np.round(self.volume_to_force(self.delta_v),5)) + 'N, total_mass = ' + str(np.round(self.mass_total,2)) + 'kg, delta_volume = ' + str(np.round(self.delta_v,5)) + 'm^3, velocity = ' + str(np.round(self.velocity[2]*yaml_p['unit_z'],2)) + 'm/s')

        global max_speed
        global min_speed
        if max_speed < self.velocity[2]:
            max_speed = self.velocity[2]
        if min_speed > self.velocity[2]:
            min_speed = self.velocity[2]

        print('min: ' + str(min_speed*yaml_p['unit_z']))
        print('max: ' + str(max_speed*yaml_p['unit_z']))

    def volume_to_force(self, delta_v):
        f = delta_v*(self.rho_air - self.rho_gas)*9.81
        return f

    def height_above_ground(self):
        return self.position[2] - self.f_terrain(self.position[0], self.position[1])[0]

    def set_ceiling(self):
        self.ceiling = random.uniform(0.9, 1) * self.size_z

    def dist_to_ceiling(self):
        return self.ceiling - self.position[2]

    def interpolate_wind(self, measurement=False):
        world = self.world_squished
        pos_z_squished = self.height_above_ground() / (self.dist_to_ceiling() + self.height_above_ground())*len(world[0,0,0,:])

        coord_x = int(np.clip(self.position[0],0,self.size_x-1))
        coord_y = int(np.clip(self.position[1],0,self.size_y-1))
        coord_z = int(np.clip(pos_z_squished,0,len(world[0,0,0,:])-1))

        x = self.position[0] - coord_x
        y = self.position[1] - coord_y
        z = pos_z_squished - coord_z

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
        if not measurement:
            w_x /= yaml_p['unit_xy']
            w_y /= yaml_p['unit_xy']
            w_z /= yaml_p['unit_z']
        return np.array([w_x, w_y, w_z, sig_xz])

    def normalize(self,x):
        return x/(abs(x)+3)
