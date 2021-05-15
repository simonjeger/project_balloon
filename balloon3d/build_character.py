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
    def __init__(self, size_x, size_y, size_z, start, target, radius_z, T, world, world_compressed):
        self.render_ratio = yaml_p['unit_xy'] / yaml_p['unit_z']
        self.radius_z = radius_z

        if yaml_p['physics']:
            self.mass = 3400 #kg
        else:
            self.mass = 1
        self.area = 21**2/4*np.pi
        self.rho = 1.2
        self.c_w = 0.45
        self.force = 10 #80 #N

        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.start = start.astype(float)
        self.target = target.astype(float)
        self.t = T
        self.action = 2

        self.world = world

        self.position = copy.copy(self.start)

        # interpolation for terrain
        x = np.linspace(0,self.size_x,len(self.world[0,:,0,0]))
        y = np.linspace(0,self.size_y,len(self.world[0,0,:,0]))
        self.f_terrain = scipy.interpolate.interp2d(x,y,self.world[0,:,:,0].T)

        self.set_ceiling()

        self.residual = self.target - self.position
        self.velocity = np.array([0,0,0])
        self.terrain = self.compress_terrain()

        if yaml_p['physics']:
            self.state = np.concatenate((self.residual.flatten(), self.velocity.flatten(), self.terrain.flatten(), world_compressed.flatten()), axis=0)
        else:
            self.state = np.concatenate((self.residual.flatten(), self.terrain.flatten(), world_compressed.flatten()), axis=0)
        self.state = self.state.astype(np.float32)

        self.path = [self.position.copy(), self.position.copy()]
        self.min_proj_dist = np.inf
        self.min_proj_dist = np.sqrt((self.residual[0]*self.render_ratio/yaml_p['radius_xy'])**2 + (self.residual[1]*self.render_ratio/yaml_p['radius_xy'])**2 + (self.residual[2]/self.radius_z)**2)

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
        self.velocity = np.array([0,0,0])
        self.terrain = self.compress_terrain()
        if yaml_p['physics']:
            self.state = np.concatenate((self.residual.flatten(), self.velocity.flatten(), self.terrain.flatten(), world_compressed.flatten()), axis=0)
        else:
            self.state = np.concatenate((self.residual.flatten(), self.terrain.flatten(), world_compressed.flatten()), axis=0)
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
        p_y = (self.path[-1][1] - self.path[-2][1])/delta_t
        p_z = (self.path[-1][2] - self.path[-2][2])/delta_t

        for _ in range(n):
            coord = [int(i) for i in np.floor(self.position)]
            in_bounds = (0 <= self.position[0] < self.size_x) & (0 <= self.position[1] < self.size_y) & (0 <= self.position[2] < self.size_z) #if still within bounds
            if in_bounds:
                # calculate velocity at time step t
                w_x, w_y, w_z, sig_xz = self.interpolate_wind()
                w_x /= yaml_p['unit_xy']
                w_y /= yaml_p['unit_xy']
                w_z /= yaml_p['unit_z']

                x = np.arange(0,self.size_x,1)
                y = np.arange(0,self.size_y,1)
                z = np.arange(0,self.size_z,1)

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

                # set velocity for state
                self.velocity = np.array([v_x, v_y, v_z])

                # write down path in history
                self.path.append(self.position.copy()) #because without copy otherwise it somehow overwrites it

                # find min_proj_dist
                self.residual = self.target - self.position
                min_proj_dist = np.sqrt((self.residual[0]*self.render_ratio/yaml_p['radius_xy'])**2 + (self.residual[1]*self.render_ratio/yaml_p['radius_xy'])**2 + (self.residual[2]/self.radius_z)**2)
                if min_proj_dist < self.min_proj_dist:
                    self.min_proj_dist = min_proj_dist

        return in_bounds

    def compress_terrain(self):
        terrain = self.world[0,:,:,0]
        pos_x = int(np.clip(self.position[0],0,self.size_x - 1))
        pos_y = int(np.clip(self.position[1],0,self.size_y - 1))

        distances = []
        indices = []
        res = 3

        init_guess = self.hight_above_ground()

        for i in range(len(terrain)*res):
            for j in range(len(terrain[0])*res):
                if np.sqrt(((i/res - self.position[0])*self.render_ratio)**2 + ((j/res - self.position[1])*self.render_ratio)**2) < init_guess:
                    distances.append(np.sqrt(((i/res - self.position[0])*self.render_ratio)**2 + ((j/res - self.position[1])*self.render_ratio)**2 + (self.f_terrain(i/res,j/res) - self.position[2])**2))
                    indices.append([i,j])

        if not distances:
            distances.append(init_guess)
            indices.append([int(self.position[0]*res),int(self.position[1]*res)])

        pos_loc_x = indices[np.argmin(distances)][0]/res
        pos_loc_y = indices[np.argmin(distances)][1]/res

        distance = np.min(distances)
        dist_x = pos_loc_x - self.position[0]
        dist_y = pos_loc_y - self.position[1]
        dist_z = self.f_terrain(pos_loc_x,pos_loc_y)[0] - self.position[2]

        other_boundaries = [self.position[0]*self.render_ratio, (self.size_x - self.position[0])*self.render_ratio, self.position[1]*self.render_ratio, (self.size_y - self.position[1])*self.render_ratio, self.ceiling[pos_x,pos_y] - self.position[2]] - distance
        case = np.argmin(other_boundaries)
        value = other_boundaries[case]
        if value < 0:
            if case == 0:
                dist_x = self.position[0]
                dist_y = 0
                dist_z = 0
            if case == 1:
                dist_x = (self.size_x - self.position[0])
                dist_y = 0
                dist_z = 0
            if case == 2:
                dist_x = 0
                dist_y = self.position[1]
                dist_z = 0
            if case == 3:
                dist_x = 0
                dist_y = (self.size_y - self.position[1])
                dist_z = 0
            if case == 4:
                dist_x = 0
                dist_y = 0
                dist_z = self.ceiling[pos_x,pos_y] - self.position[2]
        return np.array([dist_x, dist_y, dist_z])

    def hight_above_ground(self):
        return self.position[2] - self.f_terrain(self.position[0], self.position[1])[0]

    def set_ceiling(self):
        max = random.uniform(0.9, 1) * self.size_z
        self.ceiling = np.ones((self.size_x, self.size_y))*max

    def dist_to_ceiling(self):
        x = np.linspace(0,self.size_x,len(self.ceiling))
        y = np.linspace(0,self.size_y,len(self.ceiling[0]))
        f_ceiling = scipy.interpolate.interp2d(x,y,self.ceiling.T)
        return f_ceiling(self.position[0], self.position[1])[0] - self.position[2]

    def interpolate_wind(self):
        coord_x = int(self.position[0])
        coord_y = int(self.position[1])
        coord_z = int(self.position[2])

        x = self.position[0] - coord_x
        y = self.position[1] - coord_y
        z = self.position[2] - coord_z

        if coord_x == self.size_x-1:
            coord_x -= 1
            x = 1
        if coord_y == self.size_y-1:
            coord_y -= 1
            y = 1
        if coord_z == self.size_z-1:
            coord_z -= 1
            z = 1

        f_000 = self.world[-4::,coord_x,coord_y,coord_z]
        f_001 = self.world[-4::,coord_x,coord_y,coord_z+1]
        f_010 = self.world[-4::,coord_x,coord_y+1,coord_z]
        f_011 = self.world[-4::,coord_x,coord_y+1,coord_z+1]
        f_100 = self.world[-4::,coord_x+1,coord_y,coord_z]
        f_101 = self.world[-4::,coord_x+1,coord_y,coord_z+1]
        f_110 = self.world[-4::,coord_x+1,coord_y+1,coord_z]
        f_111 = self.world[-4::,coord_x+1,coord_y+1,coord_z+1]

        wind = f_000*(1-x)*(1-y)*(1-z) + f_001*(1-x)*(1-y)*z + f_010*(1-x)*y*(1-z) + f_011*(1-x)*y*z + f_100*x*(1-y)*(1-z) + f_101*x*(1-y)*z + f_110*x*y*(1-z) + f_111*x*y*z
        return wind

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
