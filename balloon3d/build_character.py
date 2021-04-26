import numpy as np
import scipy
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
    def __init__(self, size_x, size_y, size_z, start, target, T, world, world_compressed):
        self.render_ratio = yaml_p['unit_xy'] / yaml_p['unit_z']

        if yaml_p['physics']:
            self.mass = 3400 #kg
        else:
            self.mass = 1
        self.area = 21**2/4*np.pi
        self.rho = 1.2
        self.c_w = 0.45
        self.force = 80 #N

        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.start = start.astype(float)
        self.target = target.astype(float)
        self.t = T
        self.action = 2

        self.world = world

        self.position = copy.copy(self.start)

        self.set_ceiling()

        self.residual = self.target - self.position
        self.velocity = np.array([0,0,0])
        self.min_x = self.position[0] - 0
        self.max_x = self.size_x - self.position[0]
        self.min_y = self.position[1] - 0
        self.max_y = self.size_y - self.position[1]
        self.min_z = self.position[2] - 0
        self.max_z = self.dist_to_ceiling()

        if yaml_p['physics']:
            self.state = np.concatenate((self.residual.flatten(), self.velocity.flatten(), [self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z], world_compressed.flatten()), axis=0)
        else:
            self.state = np.concatenate((self.residual.flatten(), [self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z], world_compressed.flatten()), axis=0)
        self.path = [self.position.copy(), self.position.copy()]
        self.min_distance = np.sqrt((self.residual[0]*self.render_ratio)**2 + (self.residual[1]*self.render_ratio)**2 + self.residual[2]**2 + self.residual[2]**2)

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
        self.min_y = self.position[1] - 0
        self.max_y = self.size_y - self.position[1]
        self.min_z = self.position[1] - 0
        self.max_z = self.dist_to_ceiling()
        if yaml_p['physics']:
            self.state = np.concatenate((self.residual.flatten(), self.velocity.flatten(), [self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z], world_compressed.flatten()), axis=0)
        else:
            self.state = np.concatenate((self.residual.flatten(), [self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z], world_compressed.flatten()), axis=0)

        min_distance = np.sqrt((self.residual[0]*self.render_ratio)**2 + (self.residual[1]*self.render_ratio)**2 + self.residual[2]**2)
        if min_distance < self.min_distance:
            self.min_distance = min_distance

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
        p_y = (self.path[-1][1] - self.path[-2][1])/delta_t
        p_z = (self.path[-1][2] - self.path[-2][2])/delta_t

        for _ in range(n):
            coord = [int(i) for i in np.floor(self.position)]
            in_bounds = (0 <= coord[0] < self.size_x) & (0 <= coord[1] < self.size_y) & (0 <= coord[2] < self.size_z) #if still within bounds
            if in_bounds:
                # calculate velocity at time step t
                w_x = self.world[-4][coord[0], coord[1], coord[2]]/yaml_p['unit_xy'] #wind field should be in [block] and not in [m]
                w_y = self.world[-3][coord[0], coord[1], coord[2]]/yaml_p['unit_xy']
                w_z = self.world[-2][coord[0], coord[1], coord[2]]/yaml_p['unit_z']
                sig_xz = self.world[-1][coord[0], coord[1], coord[2]]

                w_x += gauss(0,sig_xz/np.sqrt(n)) #is it /sqrt(n) or just /n?
                w_y += gauss(0,sig_xz/np.sqrt(n)) #is it /sqrt(n) or just /n?
                w_z += gauss(0,sig_xz/np.sqrt(n))

                v_x = (np.sign(w_x - p_x) * (w_x - p_x)**2 * c + 0)*delta_t + p_x
                v_y = (np.sign(w_y - p_y) * (w_y - p_y)**2 * c + 0)*delta_t + p_y
                v_z = (np.sign(w_z - p_z) * (w_z - p_z)**2 * c + b)*delta_t + p_z

                # update
                self.position += [v_x*delta_t, v_y*delta_t, v_z*delta_t]
                p_x = v_x
                p_y = v_y
                p_z = v_z

                # write down path in history
                self.path.append(self.position.copy()) #because without copy otherwise it somehow overwrites it

                # set velocity for state
                self.velocity = np.array([v_x, v_y, v_z])

        return in_bounds

    def hight_above_ground(self):
        x = np.linspace(0,self.size_x,len(self.world[0,:,0,0]))
        y = np.linspace(0,self.size_y,len(self.world[0,0,:,0]))
        f = scipy.interpolate.interp2d(x,y,self.world[0,:,:,0].T)

        return self.position[2] - f(self.position[0], self.position[1])[0]

    def set_ceiling(self):
        max = random.uniform(0.9, 1) * self.size_z
        self.ceiling = np.ones((self.size_x, self.size_y))*max

    def dist_to_ceiling(self):
        x = np.linspace(0,self.size_x,len(self.ceiling))
        y = np.linspace(0,self.size_y,len(self.ceiling[0]))
        f = scipy.interpolate.interp2d(x,y,self.ceiling.T)

        return f(self.position[0], self.position[1])[0] - self.position[2]

    def become_short_sighted(self):
        sight = yaml_p['window_size']
        if yaml_p['physics']:
            self.state[0:2] = np.minimum(self.state[0], [sight]*len(self.state[0:2]))
            self.state[0:2] = np.maximum(self.state[0], [-sight]*len(self.state[0:2]))
            self.state[6:10] = np.minimum(self.state[6:10], [sight]*len(self.state[6:10]))
            self.state[6:10] = np.maximum(self.state[6:10], [-sight]*len(self.state[6:10]))
        else:
            self.state[0:2] = np.minimum(self.state[0], [sight]*len(self.state[0:2]))
            self.state[0:2] = np.maximum(self.state[0], [-sight]*len(self.state[0:2]))
            self.state[3:7] = np.minimum(self.state[3:7], [sight]*len(self.state[3:7]))
            self.state[3:7] = np.maximum(self.state[3:7], [-sight]*len(self.state[3:7]))
