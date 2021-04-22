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
    def __init__(self, size_x, size_z, start, target, T, world, world_compressed):
        if yaml_p['physics']:
            self.mass = 3400 #kg
        else:
            self.mass = 1
        self.area = 21**2/4*np.pi
        self.rho = 1.2
        self.c_w = 0.45
        self.force = 80 #N

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
        #self.min_z = self.hight_above_ground()
        self.min_z = self.position[1] - 0
        self.max_z = self.dist_to_ceiling()

        if yaml_p['physics']:
            self.state = np.concatenate((self.residual.flatten(), self.velocity.flatten(), [self.min_x, self.max_x, self.min_z, self.max_z], world_compressed.flatten()), axis=0)
        else:
            self.state = np.concatenate((self.residual.flatten(), [self.min_x, self.max_x, self.min_z, self.max_z], world_compressed.flatten()), axis=0)
        self.path = [self.position.copy(), self.position.copy()]
        self.min_distance = np.sqrt(self.residual[0]**2 + self.residual[1]**2)

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
        #self.min_z = self.hight_above_ground()
        self.min_z = self.position[1] - 0
        self.max_z = self.dist_to_ceiling()
        if yaml_p['physics']:
            self.state = np.concatenate((self.residual.flatten(), self.velocity.flatten(), [self.min_x, self.max_x, self.min_z, self.max_z], world_compressed.flatten()), axis=0)
        else:
            self.state = np.concatenate((self.residual.flatten(), [self.min_x, self.max_x, self.min_z, self.max_z], world_compressed.flatten()), axis=0)

        min_distance = np.sqrt(self.residual[0]**2 + self.residual[1]**2)
        if min_distance < self.min_distance:
            self.min_distance = min_distance

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
            coord = [int(i) for i in np.floor(self.position)]
            in_bounds = (0 <= coord[0] < self.size_x) & (0 <= coord[1] < self.size_z) #if still within bounds
            if in_bounds:
                # calculate velocity at time step t
                w_x = self.world[-3][coord[0], coord[1]]/yaml_p['unit_xy'] #wind field should be in [block] and not in [m]
                w_z = self.world[-2][coord[0], coord[1]]/yaml_p['unit_z']
                sig_xz = self.world[-1][coord[0], coord[1]]

                w_x += gauss(0,sig_xz/np.sqrt(n)) #is it /sqrt(n) or just /n?
                w_z += gauss(0,sig_xz/np.sqrt(n))

                v_x = (np.sign(w_x - p_x) * (w_x - p_x)**2 * c + 0)*delta_t + p_x
                v_z = (np.sign(w_z - p_z) * (w_z - p_z)**2 * c + b)*delta_t + p_z

                # update
                self.position += [v_x*delta_t, v_z*delta_t]
                p_x = v_x
                p_z = v_z

                # write down path in history
                self.path.append(self.position.copy()) #because without copy otherwise it somehow overwrites it

                # set velocity for state
                self.velocity = np.array([v_x, v_z])

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
