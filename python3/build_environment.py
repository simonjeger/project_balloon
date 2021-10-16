from build_character import character
from build_render import render

import pandas as pd
import matplotlib.pyplot as plt
from gym import Env, logger
from gym.spaces import Discrete, Box
import numpy as np
import torch
import scipy
import os

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

if yaml_p['environment'] == 'xplane':
    import sys
    sys.path.append('/Users/simonjeger/X-Plane 11/Resources/plugins/XPlaneConnect-1.3-rc6/Python3/src/')
    from build_character_xplane import character_xplane

logger.set_level(40) # to avoid UserWarning about box bound precision

class balloon3d(Env):
    def __init__(self, epi_n, step_n, train_or_test, writer=None):
        # which data to use
        self.train_or_test = train_or_test
        self.writer = writer
        self.radius_xy = yaml_p['radius_xy']
        self.radius_z = yaml_p['radius_z']

        # initialize state and time
        self.success_n = 0
        self.epi_n = epi_n
        self.step_n = step_n
        self.seed = 0

        # load new world to get size_x, size_z
        self.load_new_world()

        # action we can take: 0 to 1
        self.action_space = Box(low=0, high=1.0, shape=(1,), dtype=np.float64) #continuous space

        # set maximal duration of flight
        self.T = yaml_p['T']

        self.render_ratio = yaml_p['unit_xy']/yaml_p['unit_z']
        self.reset()

        # location array in x and z
        self.observation_space = Box(low=np.array([-1]*self.character.bottleneck), high=np.array([1]*self.character.bottleneck), dtype=np.float64)

        self.path_roll_out = []
        self.reward_roll_out = None
        self.reward_list = []

        self.path_reachability = []

        self.render_machine = render(self.size_x, self.size_y, self.size_z)

    def step(self, action, skip=False):
        # Interpolate the get the current world
        if yaml_p['time_dependency']:
            self.interpolate_world(self.character.t)

        coord = [int(i) for i in np.round(self.character.position_est)] #convert position into int so I can use it as index
        done = False

        # move character
        in_bounds = self.character.update(action, self.world)
        self.reward_step, done, success = self.cost(self.character.start, self.character.target, self.character.residual, self.character.U, self.character.min_proj_dist, in_bounds)
        self.success_n += success
        self.reward_epi += self.reward_step
        self.reward_list.append(self.reward_step)

        if (not skip) & (self.writer is not None):
            # logger
            if (self.step_n % yaml_p['log_frequency'] == 0) & (not done):
                self.writer.add_scalar('epi_n', self.epi_n , self.step_n)
                self.writer.add_scalar('position_x', self.character.position[0], self.step_n)
                self.writer.add_scalar('position_y', self.character.position[1], self.step_n)
                self.writer.add_scalar('position_z', self.character.position[2], self.step_n)
                self.writer.add_scalar('reward_step', self.reward_step, self.step_n)

                if yaml_p['log_world_est_error']:
                    self.writer.add_scalar('world_est_error', self.character.esterror_world, self.step_n)
            if done:
                self.writer.add_scalar('step_n', self.step_n , self.step_n)
                self.writer.add_scalar('epi_n', self.epi_n , self.step_n)
                self.writer.add_scalar('position_x', self.character.position[0], self.step_n)
                self.writer.add_scalar('position_y', self.character.position[1], self.step_n)
                self.writer.add_scalar('position_z', self.character.position[2], self.step_n)

                self.writer.add_scalar('target_x', self.character.target[0], self.step_n)
                self.writer.add_scalar('target_y', self.character.target[1], self.step_n)
                self.writer.add_scalar('target_z', self.character.target[2], self.step_n)

                self.writer.add_scalar('size_x', self.size_x , self.step_n)
                self.writer.add_scalar('size_y', self.size_y , self.step_n)
                self.writer.add_scalar('size_z', self.size_z , self.step_n)

                self.writer.add_scalar('reward_step', self.reward_step, self.step_n)
                self.writer.add_scalar('reward_epi', self.reward_epi, self.step_n)

                if yaml_p['log_world_est_error']:
                    self.writer.add_scalar('world_est_error', self.character.esterror_world, self.step_n)

                if self.reward_roll_out is not None:
                    self.writer.add_scalar('reward_epi_norm', self.reward_epi/self.reward_roll_out, self.step_n)
                else:
                    self.writer.add_scalar('reward_epi_norm', 0, self.step_n)

                self.writer.add_scalar('success_n', self.success_n, self.step_n)

            self.step_n += 1
            if done:
                self.epi_n += 1

        # set placeholder for info
        info = {}

        # return step information
        return self.character.state, self.reward_step, done, info

    def cost(self, start, target, residual, U, min_proj_dist, in_bounds):
        init_proj_min = np.sqrt(((target[0] - start[0])*self.render_ratio/self.radius_xy)**2 + ((target[1] - start[1])*self.render_ratio/self.radius_xy)**2 + ((target[2] - start[2])/self.radius_z)**2)

        if in_bounds:
            # calculate reward
            if min_proj_dist <= 1:
                reward_step = yaml_p['hit']
                success = 1
                done = True
            else:
                res = np.sqrt((residual[0]*self.render_ratio/self.radius_xy)**2 + (residual[1]*self.render_ratio/self.radius_xy)**2 + (residual[2]/self.radius_z)**2)
                reward_step = yaml_p['step']*yaml_p['delta_t'] + abs(U)*yaml_p['action'] + (init_proj_min - res)/init_proj_min*yaml_p['gradient']
                success = 0
                done = False

            if self.character.t <= 0:
                reward_step = yaml_p['overtime'] + (init_proj_min - min_proj_dist)/init_proj_min*yaml_p['min_proj_dist']
                success = 0
                done = True

        else:
            reward_step = yaml_p['bounds'] + (init_proj_min - min_proj_dist)/init_proj_min*yaml_p['min_proj_dist']
            self.character.t = 0
            success = 0
            done = True

        return reward_step, done, success

    def render(self, mode=False): #mode = False is needed so I can distinguish between when I want to render and when I don't
        self.render_machine.make_render(self.character, self.reward_step, self.reward_epi, self.world_name, self.radius_xy, self.radius_z, self.train_or_test, self.path_roll_out)

    def reset(self, target=None):
        # load new world
        if target is None:
            self.load_new_world()
            self.seed += 1 #for set_ceiling in build_character
        else:
            if yaml_p['time_dependency']:
                self.interpolate_world(yaml_p['T']) #still set back the world to time = takeoff_time

        self.reward_step = 0
        self.reward_epi = 0

        # Set problem
        if target is None:
            self.set_start()
            self.set_target()
        else:
            self.start = self.character.path[0]
            self.target = target

        # if started "under ground"
        above_ground_start = self.size_z/100
        above_ground_target = self.size_z/100
        x = np.linspace(0,self.size_x,len(self.world[0,:,0,0]))
        y = np.linspace(0,self.size_y,len(self.world[0,0,:,0]))

        f = scipy.interpolate.interp2d(x,y,self.world[0,:,:,0].T)

        if self.start[2] <= f(self.start[0], self.start[1])[0] + above_ground_start:
            self.start[2] = f(self.start[0], self.start[1])[0] + above_ground_start

        if self.target[2] <= f(self.target[0], self.target[1])[0] + above_ground_target:
            self.target[2] = f(self.target[0], self.target[1])[0] + above_ground_target

        if yaml_p['environment'] == 'python3':
            # avoid impossible szenarios
            if (self.size_z - self.start[2]) < self.size_z*yaml_p['min_space']: #a bit cheeting because the ceiling isn't in that calculation. But like this I can initialize character after the recursion.
                print('Not enough space to fly in ' + self.world_name + '. Loading new wind_map.')
                self.reset(target=target)
            self.character = character(self.size_x, self.size_y, self.size_z, self.start, self.target, self.radius_xy, self.radius_z, self.T, self.world, self.train_or_test, self.seed)
            self.reward_list = []

        elif yaml_p['environment'] == 'xplane':
            self.character = character_xplane(self.size_x, self.size_y, self.size_z, self.start, self.target, self.radius_xy, self.radius_z, self.T, self.world, self.train_or_test, self.seed)

        return self.character.state

    def load_new_world(self):
        while True:
            # choose random world_map
            if self.train_or_test == 'test':
                np.random.seed(self.seed)
                self.seed +=1
            self.world_name = np.random.choice(os.listdir(yaml_p['data_path'] + self.train_or_test + '/tensor'))

            hour = int(self.world_name[-5:-3])
            self.takeoff_time = hour*60*60 + np.random.randint(0,60)

            #if self.takeoff_time + yaml_p['T'] < 23*60*60:
            if self.takeoff_time + yaml_p['T'] < 6*60*60:
                break

        # remove suffix and timestamp
        self.world_name = self.world_name[:-6]

        # Interpolate the get the current world
        self.interpolate_world(yaml_p['T'])

        # define world size
        self.size_x = len(self.world[-1,:,:])
        self.size_y = len(self.world[-1,:,:][0])
        self.size_z = len(self.world[-1,:,:][0][0])

    def interpolate_world(self,t):
        tss = self.takeoff_time + yaml_p['T'] - t #time since start

        h = int(tss/60/60)
        p = (tss - h*60*60)/60/60
        self.world_0 = torch.load(yaml_p['data_path'] + self.train_or_test + '/tensor/' + self.world_name + '_'  + str(h).zfill(2) + '.pt')
        self.world_1 = torch.load(yaml_p['data_path'] + self.train_or_test + '/tensor/' + self.world_name + '_'  + str(h+1).zfill(2) + '.pt')

        self.world = p*(self.world_1 - self.world_0) + self.world_0
        torch.save(self.world, yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/render/world.pt')

    def set_start(self):
        if self.train_or_test == 'train':
            yaml_start = yaml_p['start_train']
        else:
            yaml_start = yaml_p['start_test']
            np.random.seed(self.seed)
            self.seed +=1

        if (yaml_start == 'center') | (yaml_start == 'center_determ'):
            self.start = np.array([(self.size_x - 1)/2,(self.size_y - 1)/2,0], dtype=float)
        else:
            self.start = np.array(yaml_start, dtype=float)
        if yaml_start != 'center_determ':
            self.start = np.floor(self.start) + np.append(np.random.uniform(-1,1,2),[0]) #randomize start position without touching the z-axis

    def set_target(self):
        border_x = self.size_x/(10*self.render_ratio)
        border_y = self.size_y/(10*self.render_ratio)
        border_z = self.size_z/10

        if self.train_or_test == 'train':
            yaml_target = yaml_p['target_train']
        else:
            yaml_target = yaml_p['target_test']
            np.random.seed(self.seed)
            self.seed +=1

        if yaml_target == 'random':
            self.target = np.array([border_x + np.random.uniform()*(self.size_x - 2*border_x),border_y + np.random.uniform()*(self.size_y - 2*border_y),border_z + np.random.uniform()*(self.size_z - 2*border_z)], dtype=float)
        elif yaml_target == 'random_low':
            self.target = np.array([border_x + np.random.uniform()*(self.size_x - 2*border_x),border_y + np.random.uniform()*(self.size_y - 2*border_y),0], dtype=float)
        elif yaml_target == 'right':
            self.target = np.array([self.start[0] + np.random.uniform()*(self.size_x - self.start[0] - border_x),border_y + np.random.uniform()*(self.size_y - 2*border_y),border_z + np.random.uniform()*(self.size_z - 2*border_z)], dtype=float)
        elif yaml_target == 'right_low':
            self.target = np.array([self.start[0] + np.random.uniform()*(self.size_x - self.start[0] - border_x),border_y + np.random.uniform()*(self.size_y - 2*border_y),0], dtype=float)
        else:
            self.target = np.array(yaml_target, dtype=float)
