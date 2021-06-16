from build_render import build_render
from human_autoencoder import HAE
from build_autoencoder import VAE
from build_character import character

import pandas as pd
import matplotlib.pyplot as plt
from gym import Env, logger
from gym.spaces import Discrete, Box
import numpy as np
import random
import torch
from random import gauss
import os

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

logger.set_level(40) # to avoid UserWarning about box bound precision

class balloon2d(Env):
    def __init__(self, epi_n, step_n, train_or_test, writer=None, radius_x=yaml_p['radius_stop_x'], radius_z=yaml_p['radius_stop_x']):
        # which data to use
        self.train_or_test = train_or_test
        self.writer = writer
        self.radius_x = radius_x
        self.radius_z = radius_z

        # initialize state and time
        self.success_n = 0
        self.epi_n = epi_n
        self.step_n = step_n
        self.seed = 0

        # initialize autoencoder object
        if yaml_p['autoencoder'][0:3] == 'HAE':
            self.ae = HAE()
        if yaml_p['autoencoder'] == 'VAE':
            self.ae = VAE()

        # load new world to get size_x, size_z
        self.load_new_world()

        if yaml_p['continuous']:
            # action we can take: -1 to 1
            self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64) #continuous space
        else:
            # action we can take: down, stay, up
            self.action_space = Discrete(3) #discrete space

        # set maximal duration of flight
        self.T = yaml_p['T']

        self.render_ratio = yaml_p['unit_x']/yaml_p['unit_z']
        self.reset()

        # location array in x and z
        regular_state_space_low = np.array([-1]*(2+2+self.character.bottleneck+2)) #residual, velocity, boundaries, measurement
        regular_state_space_high = np.array([1]*(2+2+self.character.bottleneck+2))
        world_compressed_state_space_low = np.array([-1]*self.ae.bottleneck)
        world_compressed_state_space_high = np.array([1]*self.ae.bottleneck)
        self.observation_space = Box(low=np.concatenate((regular_state_space_low, world_compressed_state_space_low), axis=0), high=np.concatenate((regular_state_space_high, world_compressed_state_space_high), axis=0), dtype=np.float64) #ballon_x = [0,...,100], balloon_z = [0,...,30], error_x = [0,...,100], error_z = [0,...,30]

        self.path_roll_out = None
        self.reward_roll_out = None
        self.reward_list = []

    def step(self, action, roll_out=False):
        # Update compressed wind map
        if self.prev_int != int(self.character.position[0]):
            self.world_compressed = self.ae.compress(self.world, self.character.position, self.character.ceiling)
            self.prev_int = int(self.character.position[0])

        coord = [int(i) for i in np.round(self.character.position)] #convert position into int so I can use it as index
        done = False

        # move character
        in_bounds = self.character.update(action, self.world_compressed, roll_out)
        done = self.cost(in_bounds)

        if not roll_out:
            # logger
            if self.writer is not None:
                if (self.step_n % yaml_p['log_frequency'] == 0) & (not done):
                    self.writer.add_scalar('epi_n', self.epi_n , self.step_n)
                    self.writer.add_scalar('position_x', self.character.position[0], self.step_n)
                    self.writer.add_scalar('position_z', self.character.position[1], self.step_n)
                    self.writer.add_scalar('reward_step', self.reward_step, self.step_n)
                if done:
                    self.writer.add_scalar('step_n', self.step_n , self.step_n)
                    self.writer.add_scalar('epi_n', self.epi_n , self.step_n)
                    self.writer.add_scalar('position_x', self.character.position[0], self.step_n)
                    self.writer.add_scalar('position_z', self.character.position[1], self.step_n)

                    self.writer.add_scalar('target_x', self.character.target[0], self.step_n)
                    self.writer.add_scalar('target_z', self.character.target[1], self.step_n)

                    self.writer.add_scalar('size_x', self.size_x , self.step_n)
                    self.writer.add_scalar('size_z', self.size_z , self.step_n)

                    self.writer.add_scalar('radius_x', self.radius_x , self.step_n)
                    self.writer.add_scalar('radius_z', self.radius_z , self.step_n)

                    self.writer.add_scalar('reward_step', self.reward_step, self.step_n)
                    self.writer.add_scalar('reward_epi', self.reward_epi, self.step_n)
                    self.writer.add_scalar('reward_epi_norm', self.reward_epi/self.reward_roll_out, self.step_n)

                    self.writer.add_scalar('success_n', self.success_n , self.step_n)

            self.step_n += 1
            if done:
                self.epi_n += 1

        # set placeholder for info
        info = {}

        # return step information
        return self.character.state, self.reward_step, done, info

    def cost(self, in_bounds):
        init_proj_min = np.sqrt(((self.character.target[0] - self.character.start[0])*self.render_ratio/self.radius_x)**2 + ((self.character.target[1] - self.character.start[1])/self.radius_z)**2)

        if in_bounds:
            # calculate reward
            if self.character.min_proj_dist <= 1:
                self.reward_step = yaml_p['hit']
                self.success_n += 1
                done = True
            else:
                if yaml_p['type'] == 'regular':
                    self.reward_step = yaml_p['step']*yaml_p['time'] + abs(self.character.action - 1)*yaml_p['action']
                elif yaml_p['type'] == 'squished':
                    self.reward_step = yaml_p['step']*yaml_p['time'] + abs(self.character.U)*yaml_p['action']
                done = False

            if self.character.t <= 0:
                self.reward_step = yaml_p['overtime'] + (init_proj_min - self.character.min_proj_dist)/init_proj_min * yaml_p['min_proj_dist']
                done = True

        else:
            self.reward_step = yaml_p['bounds'] + (init_proj_min - self.character.min_proj_dist)/init_proj_min * yaml_p['min_proj_dist']
            self.character.t = 0
            done = True

        self.reward_epi += self.reward_step
        self.reward_list.append(self.reward_step)
        return done

    def render(self, mode=False): #mode = False is needed so I can distinguish between when I want to render and when I don't
        build_render(self.character, self.reward_step, self.reward_epi, self.world_name, self.ae.window_size, self.radius_x, self.radius_z, self.train_or_test, self.path_roll_out)

    def reset(self, roll_out=False):
        if not roll_out:
            # load new world
            self.load_new_world()

        self.reward_step = 0
        self.reward_epi = 0

        # Set problem
        if not roll_out:
            start = self.set_start()
        else:
            start = self.character.path[0]
        target = self.set_target(start)

        # if started "under ground"
        above_ground_start = self.size_z/100
        above_ground_target = self.size_z/5
        x = np.linspace(0,self.size_x,len(self.world[0,:,0]))

        if start[1] <= np.interp(start[0],x,self.world[0,:,0]) + above_ground_start:
            start[1] = np.ceil(np.interp(start[0],x,self.world[0,:,0]) + above_ground_start)

        if target[1] <= np.interp(target[0],x,self.world[0,:,0]) + above_ground_target:
            target[1] = np.interp(target[0],x,self.world[0,:,0]) + above_ground_target

        # Initial compressed wind map
        self.world_compressed = self.ae.compress(self.world, start, self.size_z)

        self.character = character(self.size_x, self.size_z, start, target, self.radius_x, self.radius_z, self.T, self.world, self.world_compressed)

        # avoid impossible szenarios
        min_space = self.size_z*yaml_p['min_space']
        pos_x = int(np.clip(self.character.position[0],0,self.size_x - 1))
        if self.character.ceiling - self.world[0,pos_x,0] < min_space:
            self.reset()

        self.reward_list = []
        self.prev_int = -1

        return self.character.state

    def load_new_world(self):
        # choose random world_map
        if self.train_or_test == 'test':
            random.seed(self.seed)
            self.seed +=1
        self.world_name = random.choice(os.listdir(yaml_p['data_path'] + self.train_or_test + '/tensor'))

        # remove suffix
        length = len(self.world_name)
        self.world_name = self.world_name[:length - 3]
        # read in world_map
        self.world = torch.load(yaml_p['data_path'] + self.train_or_test + '/tensor/' + self.world_name + '.pt')

        # define world size
        self.size_x = len(self.world[-1,:,:])
        self.size_z = len(self.world[-1,:,:][0])

    def set_start(self):
        border_x = self.size_x/(10*self.render_ratio)
        border_z = self.size_z/10

        if self.train_or_test == 'train':
            yaml_start = yaml_p['start_train']
        else:
            yaml_start = yaml_p['start_test']

        if yaml_start == 'random':
            start = np.array([border_x + random.random()*(self.size_x - 2*border_x),border_z + random.random()*(self.size_z - 2*border_z)], dtype=float)
        elif yaml_start == 'random_low':
            start = np.array([border_x + random.random()*(self.size_x - 2*border_x),0], dtype=float)
        elif yaml_start == 'left':
            start = np.array([border_x + random.random()*(self.size_x/2 - 2*border_x),border_z + random.random()*(self.size_z - 2*border_z)], dtype=float)
        elif yaml_start == 'left_low':
            start = np.array([border_x + random.random()*(self.size_x/2 - 2*border_x),0], dtype=float)
        else:
            start = np.array(yaml_start, dtype=float)
        return start

    def set_target(self, start):
        border_x = self.size_x/(10*self.render_ratio)
        border_z = self.size_z/10

        if self.train_or_test == 'train':
            yaml_target = yaml_p['target_train']
        else:
            yaml_target = yaml_p['target_test']

        if yaml_target == 'random':
            target = np.array([border_x + random.random()*(self.size_x - 2*border_x),border_z + random.random()*(self.size_z - 2*border_z)], dtype=float)
        elif yaml_target == 'random_low':
            target = np.array([border_x + random.random()*(self.size_x - 2*border_x),0], dtype=float)
        elif yaml_target == 'right':
            target = np.array([start[0] + random.random()*(self.size_x - start[0] - border_x),border_z + random.random()*(self.size_z - 2*border_z)], dtype=float)
        elif yaml_target == 'right_low':
            target = np.array([start[0] + random.random()*(self.size_x - start[0] - border_x),0], dtype=float)
        else:
            target = np.array(yaml_target, dtype=float)

        return target
