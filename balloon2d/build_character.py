import numpy as np
import random
from random import gauss
import copy

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

class character():
    def __init__(self, size_x, size_z, start, target, radius_z, T, world, world_compressed):
        self.render_ratio = yaml_p['unit_xy'] / yaml_p['unit_z']
        self.radius_z = radius_z

        if yaml_p['physics']:
            self.mass = 3400 #kg
        else:
            self.mass = 1
        self.area = 21**2/4*np.pi
        self.rho = 1.2
        self.c_w = 0.45
        self.force = 80 #80 #N

        self.size_x = size_x
        self.size_z = size_z
        self.start = start.astype(float)
        self.target = target.astype(float)
        self.t = T
        self.action = 2

        self.world = world

        self.position = copy.copy(self.start)

        self.set_ceiling()

        self.residual = self.target - self.position
        self.velocity = np.array([0,0])
        self.min_x = self.position[0] - 0
        self.max_x = self.size_x - self.position[0]
        self.min_z = self.position[1] - 0
        self.max_z = self.dist_to_ceiling()

        if yaml_p['physics']:
            self.state = np.concatenate((self.residual.flatten(), self.velocity.flatten(), [self.min_x, self.max_x, self.min_z, self.max_z], world_compressed.flatten()), axis=0)
        else:
            self.state = np.concatenate((self.residual.flatten(), [self.min_x, self.max_x, self.min_z, self.max_z], world_compressed.flatten()), axis=0)
        self.state = self.state.astype(np.float32)

        self.path = [self.position.copy(), self.position.copy()]
        self.min_proj_dist = np.sqrt((self.residual[0]*self.render_ratio/yaml_p['radius_x'])**2 + (self.residual[1]/self.radius_z)**2)

        if yaml_p['short_sighted']:
            self.become_short_sighted()

    def update(self, action, world_compressed):
        self.action = action

        in_bounds = self.move_particle(100)
        if self.hight_above_ground() < 0: # check if crashed into terrain
            in_bounds = False
        if self.dist_to_ceiling() < 0: # check if crashed into terrain
            in_bounds = False

        # update state
        self.residual = self.target - self.position
        self.min_x = self.position[0] - 0
        self.max_x = self.size_x - self.position[0]
        self.min_z = self.position[1] - 0
        self.max_z = self.dist_to_ceiling()
        if yaml_p['physics']:
            self.state = np.concatenate((self.residual.flatten(), self.velocity.flatten(), [self.min_x, self.max_x, self.min_z, self.max_z], world_compressed.flatten()), axis=0)
        else:
            self.state = np.concatenate((self.residual.flatten(), [self.min_x, self.max_x, self.min_z, self.max_z], world_compressed.flatten()), axis=0)
        self.state = self.state.astype(np.float32)

        if yaml_p['short_sighted']:
            self.become_short_sighted()

        # reduce flight length by 1 second
        self.t -= 1

        return in_bounds

    def move_particle(self, n):
        c = self.area*self.rho*self.c_w/(2*self.mass)
        b = (self.action - 1)*self.force/yaml_p['unit_z']/self.mass
        delta_t = yaml_p['time']/n

        p_x = (self.path[-1][0] - self.path[-2][0])/delta_t
        p_z = (self.path[-1][1] - self.path[-2][1])/delta_t

        for _ in range(n):
            in_bounds = (0 <= self.position[0] < self.size_x) & (0 <= self.position[1] < self.size_z) #if still within bounds
            if in_bounds:
                # calculate velocity at time step t
                w_x, w_z, sig_xz = self.interpolate_wind()
                w_x /= yaml_p['unit_xy']
                w_z /= yaml_p['unit_z']

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
                min_proj_dist = np.sqrt((self.residual[0]*self.render_ratio/yaml_p['radius_x'])**2 + (self.residual[1]/self.radius_z)**2)
                if min_proj_dist < self.min_proj_dist:
                    self.min_proj_dist = min_proj_dist

        return in_bounds

    def hight_above_ground(self):
        x = np.linspace(0,self.size_x,len(self.world[0,:,0]))
        return self.position[1] - np.interp(self.position[0],x,self.world[0,:,0])

    def set_ceiling(self):
        max = random.uniform(0.9, 1) * self.size_z
        self.ceiling = [max]*self.size_x

    def dist_to_ceiling(self):
        x = np.linspace(0,self.size_x,len(self.ceiling))
        return np.interp(self.position[0],x,self.ceiling) - self.position[1]

    def interpolate_wind(self):
        coord_x = int(self.position[0])
        coord_z = int(self.position[1])

        x = self.position[0] - coord_x
        z = self.position[1] - coord_z

        if coord_x == self.size_x-1:
            coord_x -= 1
            x = 1
        if coord_z == self.size_z-1:
            coord_z -= 1
            z = 1

        f_00 = self.world[-3::,coord_x,coord_z]
        f_10 = self.world[-3::,coord_x+1,coord_z]
        f_01 = self.world[-3::,coord_x,coord_z+1]
        f_11 = self.world[-3::,coord_x+1,coord_z+1]

        wind = f_00*(1-x)*(1-z) + f_10*x*(1-z) + f_01*(1-x)*z + f_11*x*z
        return wind

    def become_short_sighted(self):
        sight = yaml_p['window_size']
        if yaml_p['physics']:
            self.state[0] = np.minimum(self.state[0], sight)
            self.state[0] = np.maximum(self.state[0], -sight)
            self.state[4:6] = np.minimum(self.state[4:6], [sight]*len(self.state[4:6]))
            self.state[4:6] = np.maximum(self.state[4:6], [-sight]*len(self.state[4:6]))
        else:
            self.state[0] = np.minimum(self.state[0], sight)
            self.state[0] = np.maximum(self.state[0], -sight)
            self.state[2:4] = np.minimum(self.state[2:4], [sight]*len(self.state[2:4]))
            self.state[2:4] = np.maximum(self.state[2:4], [-sight]*len(self.state[2:4]))
