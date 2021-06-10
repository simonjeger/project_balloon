import numpy as np
import random
from random import gauss
import copy

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

class character():
    def __init__(self, size_x, size_z, start, target, radius_x, radius_z, T, world, world_compressed):
        self.render_ratio = yaml_p['unit_x'] / yaml_p['unit_z']
        self.radius_x = radius_x
        self.radius_z = radius_z

        if yaml_p['balloon'] == 'big':
            self.mass = 3400 #kg
            self.area = 21**2/4*np.pi
            self.rho = 1.2
            self.c_w = 0.45
            self.force = 80 #80 #N
        elif yaml_p['balloon'] == 'small':
            self.mass = 3 #kg
            self.area = 2.3**2/4*np.pi
            self.rho = 1.2
            self.c_w = 0.45
            self.force = 0.2 #N
        else:
            print('ERROR: Please choose a balloon')

        self.size_x = size_x
        self.size_z = size_z
        self.start = start.astype(float)
        self.target = target.astype(float)
        self.t = T
        self.action = 2

        self.world = world
        self.world_compressed = world_compressed

        self.position = copy.copy(self.start)
        self.velocity = np.array([0,0])

        self.set_ceiling()
        self.world_squished = squish(self.world, self.ceiling)

        self.residual = self.target - self.position
        self.measurement = self.interpolate_wind(measurement=True)[0:2]

        self.set_state()

        self.path = [self.position.copy(), self.position.copy()]
        self.min_proj_dist = np.sqrt((self.residual[0]*self.render_ratio/self.radius_x)**2 + (self.residual[1]/self.radius_z)**2)

        if yaml_p['short_sighted']:
            self.become_short_sighted()

    def update(self, action, world_compressed, roll_out=False):
        self.action = action
        self.world_compressed = world_compressed

        in_bounds = self.move_particle(int(yaml_p['time']/6), roll_out)
        if self.height_above_ground() < 0: #check if crashed into terrain
            in_bounds = False
        if self.dist_to_ceiling() < 0: #check if crashed into ceiling
            in_bounds = False

        self.residual = self.target - self.position
        self.measurement = self.interpolate_wind(measurement=True)[0:2]

        self.set_state()

        if yaml_p['short_sighted']:
            self.become_short_sighted()

        # reduce flight length by 1 second
        self.t -= 1

        return in_bounds

    def set_state(self):
        if yaml_p['type'] == 'regular':
            if yaml_p['boundaries'] == 'short':
                boundaries = np.concatenate((self.compress_terrain()/[self.size_x, self.size_z], [self.position[0] - np.floor(self.position[0])]))
                self.bottleneck = len(boundaries)
            elif yaml_p['boundaries'] == 'long':
                min_x = self.position[0]/self.size_x
                max_x = (self.size_x - self.position[0])/self.size_x
                min_z = self.position[1]/self.size_z
                max_z = self.dist_to_ceiling()/self.size_z
                boundaries = np.array([min_x, max_x, min_z, max_z, self.height_above_ground()])
                self.bottleneck = len(boundaries)
            self.state = np.concatenate(((self.residual/[self.size_x,self.size_z]).flatten(), self.normalize(self.velocity.flatten()), boundaries.flatten(), self.normalize(self.measurement.flatten()), self.normalize(self.world_compressed.flatten())), axis=0)

        elif yaml_p['type'] == 'squished':
            tar_x = int(np.clip(self.target[0],0,self.size_x - 1))
            self.res_z_squished = (self.target[1]-self.world[0,tar_x,0])/(self.ceiling - self.world[0,tar_x,0]) - self.height_above_ground() / (self.dist_to_ceiling() + self.height_above_ground())

            if yaml_p['boundaries'] == 'short':
                boundaries = np.array([self.position[0] - np.floor(self.position[0])])
                self.bottleneck = len(boundaries)

            elif yaml_p['boundaries'] == 'long':
                min_x = self.position[0]/self.size_x
                max_x = (self.size_x - self.position[0])/self.size_x
                min_z = self.position[1]/self.size_z
                max_z = self.dist_to_ceiling()/self.size_z
                boundaries = np.array([min_x, max_x, min_z, max_z])
                self.bottleneck = len(boundaries)

            self.state = np.concatenate(([self.residual[0]/self.size_x], [self.res_z_squished], self.normalize(self.velocity.flatten()), boundaries.flatten(), self.normalize(self.measurement.flatten()), self.normalize(self.world_compressed.flatten())), axis=0)
        self.state = self.state.astype(np.float32)

    def move_particle(self, n, roll_out):
        c = self.area*self.rho*self.c_w/(2*self.mass)
        delta_t = yaml_p['time']/n

        p_x = (self.path[-1][0] - self.path[-2][0])/delta_t
        p_z = (self.path[-1][1] - self.path[-2][1])/delta_t

        self.U = 0
        for _ in range(n):
            if (yaml_p['type'] == 'regular') & (not roll_out):
                b = (self.action - 1)*self.force/yaml_p['unit_z']/self.mass

            else:
                dist_bottom = self.height_above_ground()
                dist_top = self.dist_to_ceiling()
                rel_pos = dist_bottom / (dist_top + dist_bottom)
                velocity = self.velocity[1]
                u = ll_pd(self.action,rel_pos,velocity)

                b = u*self.force/yaml_p['unit_z']/self.mass
                self.U += abs(u)/n

            in_bounds = (0 <= self.position[0] < self.size_x) & (0 <= self.position[1] < self.size_z) #if still within bounds
            if in_bounds:
                # calculate velocity at time step t
                w_x, w_z, sig_xz = self.interpolate_wind()

                w_x += gauss(0,sig_xz/np.sqrt(n)) #is it /sqrt(n) or just /n?
                w_z += gauss(0,sig_xz/np.sqrt(n))

                v_x = (np.sign(w_x - p_x) * (w_x - p_x)**2 * c + 0)*delta_t + p_x
                v_z = (np.sign(w_z - p_z) * (w_z - p_z)**2 * c + b)*delta_t + p_z

                # update
                self.position += [v_x*delta_t, v_z*delta_t]
                p_x = v_x
                p_z = v_z

                # set velocity for state
                self.velocity = np.array([v_x, v_z])

                # write down path in history
                self.path.append(self.position.copy()) #because without copy otherwise it somehow overwrites it

                # find min_proj_dist
                self.residual = self.target - self.position
                min_proj_dist = np.sqrt((self.residual[0]*self.render_ratio/self.radius_x)**2 + (self.residual[1]/self.radius_z)**2)
                if min_proj_dist < self.min_proj_dist:
                    self.min_proj_dist = min_proj_dist

        return in_bounds

    def compress_terrain(self):
        terrain = self.world[0,:,0]
        pos_x = int(np.clip(self.position[0],0,self.size_x - 1))

        x = np.linspace(0,self.size_x,len(terrain))
        distances = []
        res = 10
        for i in range(len(terrain)*res):
            distances.append(np.sqrt(((i/res - self.position[0])*self.render_ratio)**2 + (np.interp(i/res,x,terrain) - self.position[1])**2))

        distance = np.min(distances)
        dist_x = np.argmin(distances)/res - self.position[0]
        dist_z = np.interp(np.argmin(distances)/res,x,terrain) - self.position[1]

        other_boundaries = [self.position[0]*self.render_ratio, (self.size_x - self.position[0])*self.render_ratio, self.ceiling - self.position[1]] - distance
        case = np.argmin(other_boundaries)
        value = other_boundaries[case]

        if value < 0:
            if case == 0:
                dist_x = self.position[0]
                dist_z = 0
            if case == 1:
                dist_x = self.size_x - self.position[0]
                dist_z = 0
            if case == 2:
                dist_x = 0
                dist_z = self.ceiling - self.position[1]

        return np.array([dist_x, dist_z])

    def height_above_ground(self):
        x = np.linspace(0,self.size_x,len(self.world[0,:,0]))
        return self.position[1] - np.interp(self.position[0],x,self.world[0,:,0])

    def set_ceiling(self):
        self.ceiling = random.uniform(0.9, 1) * self.size_z

    def dist_to_ceiling(self):
        return self.ceiling - self.position[1]

    def interpolate_wind(self, measurement=False):
        world = self.world_squished

        pos_z_squished = self.height_above_ground() / (self.dist_to_ceiling() + self.height_above_ground())*len(world[0,0,:])

        coord_x = int(np.clip(self.position[0],0,self.size_x-1))
        coord_z = int(np.clip(pos_z_squished,0,len(world[0,0,:])-1))

        x = self.position[0] - coord_x
        z = pos_z_squished - coord_z

        if coord_x == self.size_x-1:
            coord_x -= 1
            x = 1
        if coord_z == self.size_z-1:
            coord_z -= 1
            z = 1

        f_00 = world[-3::,coord_x,coord_z]
        f_10 = world[-3::,coord_x+1,coord_z]
        f_01 = world[-3::,coord_x,coord_z+1]
        f_11 = world[-3::,coord_x+1,coord_z+1]

        wind = f_00*(1-x)*(1-z) + f_10*x*(1-z) + f_01*(1-x)*z + f_11*x*z

        w_x, w_z, sig_xz = wind
        if not measurement:
            w_x /= yaml_p['unit_x']
            w_z /= yaml_p['unit_z']
        return np.array([w_x, w_z, sig_xz])

    def normalize(self,x):
        return x/(abs(x)+3)

    def become_short_sighted(self):
        sight = yaml_p['window_size']
        self.state[0] = np.minimum(self.state[0], sight)
        self.state[0] = np.maximum(self.state[0], -sight)
        self.state[4:6] = np.minimum(self.state[4:6], [sight]*len(self.state[4:6]))
        self.state[4:6] = np.maximum(self.state[4:6], [-sight]*len(self.state[4:6]))
