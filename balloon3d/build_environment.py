from build_render import build_render
from human_autoencoder import HAE
from build_autoencoder import VAE
from build_character import character

import pandas as pd
import matplotlib.pyplot as plt
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import torch
import scipy
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


class balloon2d(Env):
    def __init__(self, epi_n, step_n, train_or_test, writer=None, radius_xy=yaml_p['radius_stop_xy'], radius_z=yaml_p['radius_stop_xy']):
        # which data to use
        self.train_or_test = train_or_test
        self.writer = writer
        self.radius_xy = radius_xy
        self.radius_z = radius_z

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

        self.render_ratio = yaml_p['unit_xy']/yaml_p['unit_z']
        self.reset()

        # location array in x and z
        if yaml_p['type'] == 'regular':
            if yaml_p['physics']:
                regular_state_space_low = np.array(np.concatenate(([-self.size_x,-self.size_y,-self.size_z,-np.inf,-np.inf,-np.inf],[-np.inf]*self.character.bottleneck,[-np.inf]*3))) #residual to target, velocity, measurement, distance to border
                regular_state_space_high = np.array(np.concatenate(([self.size_x,self.size_y,self.size_z,np.inf,np.inf,np.inf],[np.inf]*self.character.bottleneck,[np.inf]*3)))
            else:
                regular_state_space_low = np.array(np.concatenate(([-self.size_x,-self.size_y,-self.size_z],[-np.inf]*self.character.bottleneck,[-np.inf]*3))) #residual to target, velocity, measurement, distance to border
                regular_state_space_high = np.array(np.concatenate(([self.size_x,self.size_y,self.size_z],[np.inf]*self.character.bottleneck,[np.inf]*3)))
            world_compressed_state_space_low = np.array([-1]*self.ae.bottleneck)
            world_compressed_state_space_high = np.array([1]*self.ae.bottleneck)
            self.observation_space = Box(low=np.concatenate((regular_state_space_low, world_compressed_state_space_low), axis=0), high=np.concatenate((regular_state_space_high, world_compressed_state_space_high), axis=0), dtype=np.float64) #ballon_x = [0,...,100], balloon_z = [0,...,30], error_x = [0,...,100], error_z = [0,...,30]

        elif yaml_p['type'] == 'squished':
            if yaml_p['physics']:
                regular_state_space_low = np.array(np.concatenate(([0,0,0,-np.inf,-np.inf,-np.inf],[-np.inf]*self.character.bottleneck,[-np.inf]*3))) #residual to target, velocity, measurement
                regular_state_space_high = np.array(np.concatenate(([self.size_x,self.size_y,1,np.inf,np.inf,np.inf],[np.inf]*self.character.bottleneck,[np.inf]*3)))
            else:
                regular_state_space_low = np.array(np.concatenate(([0,0,0],[-np.inf]*self.character.bottleneck,[-np.inf]*3))) #residual to target, measurement
                regular_state_space_high = np.array(np.concatenate(([self.size_x,self.size_y,1],[np.inf]*self.character.bottleneck,[np.inf]*3)))
            world_compressed_state_space_low = np.array([-1]*self.ae.bottleneck)
            world_compressed_state_space_high = np.array([1]*self.ae.bottleneck)
            self.observation_space = Box(low=np.concatenate((regular_state_space_low, world_compressed_state_space_low), axis=0), high=np.concatenate((regular_state_space_high, world_compressed_state_space_high), axis=0), dtype=np.float64) #ballon_x = [0,...,100], balloon_z = [0,...,30], error_x = [0,...,100], error_z = [0,...,30]

        # initialize state and time
        self.success_n = 0
        self.epi_n = epi_n
        self.step_n = step_n

        self.path_roll_out = None

    def step(self, action, roll_out=False):
        # Update compressed wind map
        self.world_compressed = self.ae.compress(self.world, self.character.position, self.character.ceiling)

        coord = [int(i) for i in np.round(self.character.position)] #convert position into int so I can use it as index
        done = False

        # move character
        in_bounds = self.character.update(action, self.world_compressed)
        done = self.cost(in_bounds)

        if not roll_out:
            # logger
            if self.writer is not None:
                if (self.step_n % yaml_p['log_frequency'] == 0) & (not done):
                    self.writer.add_scalar('episode', self.epi_n , self.step_n)
                    self.writer.add_scalar('position_x', self.character.position[0], self.step_n)
                    self.writer.add_scalar('position_y', self.character.position[1], self.step_n)
                    self.writer.add_scalar('position_z', self.character.position[2], self.step_n)
                    self.writer.add_scalar('reward_step', self.reward_step, self.step_n)
                if done:
                    self.writer.add_scalar('episode', self.epi_n , self.step_n)
                    self.writer.add_scalar('position_x', self.character.position[0], self.step_n)
                    self.writer.add_scalar('position_y', self.character.position[1], self.step_n)
                    self.writer.add_scalar('position_z', self.character.position[2], self.step_n)
                    self.writer.add_scalar('reward_step', self.reward_step, self.step_n)

                    self.writer.add_scalar('size_x', self.size_x , self.step_n)
                    self.writer.add_scalar('size_y', self.size_y , self.step_n)
                    self.writer.add_scalar('size_z', self.size_z , self.step_n)
                    self.writer.add_scalar('target_x', self.character.target[0], self.step_n)
                    self.writer.add_scalar('target_y', self.character.target[1], self.step_n)
                    self.writer.add_scalar('target_z', self.character.target[2], self.step_n)
                    self.writer.add_scalar('reward_epi', self.reward_epi, self.step_n)

                    self.writer.add_scalar('success_n', self.success_n, self.step_n)

            self.step_n += 1
            if done:
                self.epi_n += 1

        # set placeholder for info
        info = {}

        # return step information
        return self.character.state, self.reward_step, done, info

    def cost(self, in_bounds):
        init_proj_min = np.sqrt(((self.character.target[0] - self.character.start[0])*self.render_ratio/self.radius_xy)**2 + ((self.character.target[1] - self.character.start[1])*self.render_ratio/self.radius_xy)**2 + ((self.character.target[2] - self.character.start[2])/self.radius_z)**2)

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
        return done

    def render(self, mode=False): #mode = False is needed so I can distinguish between when I want to render and when I don't
        build_render(self.character, self.reward_step, self.reward_epi, self.world_name, self.ae.window_size, self.radius_xy, self.radius_z, self.train_or_test, self.path_roll_out)

    def reset(self, roll_out=False):
        # load new world
        if not roll_out:
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
        x = np.linspace(0,self.size_x,len(self.world[0,:,0,0]))
        y = np.linspace(0,self.size_y,len(self.world[0,0,:,0]))

        f = scipy.interpolate.interp2d(x,y,self.world[0,:,:,0].T)

        if start[2] <= f(start[0], start[1])[0] + above_ground_start:
            start[2] = f(start[0], start[1])[0] + above_ground_start

        if target[2] <= f(target[0], target[1])[0] + above_ground_target:
            target[2] = f(target[0], target[1])[0] + above_ground_target

        # Initial compressed wind map
        self.world_compressed = self.ae.compress(self.world, start, np.ones((self.size_x,self.size_y))*self.size_z)

        self.character = character(self.size_x, self.size_y, self.size_z, start, target, self.radius_xy, self.radius_z, self.T, self.world, self.world_compressed)

        # avoid impossible szenarios
        min_space = self.size_z/5
        pos_x = int(np.clip(self.character.position[0],0,self.size_x - 1))
        pos_y = int(np.clip(self.character.position[1],0,self.size_y - 1))
        if self.character.ceiling[pos_x,pos_y] - self.world[0,pos_x,pos_y,0] < min_space:
            self.reset()
        return self.character.state

    def load_new_world(self):
        # choose random world_map
        self.world_name = random.choice(os.listdir(yaml_p['data_path'] + self.train_or_test + '/tensor'))
        # remove suffix
        length = len(self.world_name)
        self.world_name = self.world_name[:length - 3]
        # read in world_map
        self.world = torch.load(yaml_p['data_path'] + self.train_or_test + '/tensor/' + self.world_name + '.pt')

        # define world size
        self.size_x = len(self.world[-1,:,:])
        self.size_y = len(self.world[-1,:,:][0])
        self.size_z = len(self.world[-1,:,:][0][0])

    def set_start(self):
        border_x = self.size_x/(10*self.render_ratio)
        border_y = self.size_y/(10*self.render_ratio)
        border_z = self.size_z/10

        if self.train_or_test == 'train':
            yaml_start = yaml_p['start_train']
        else:
            yaml_start = yaml_p['start_test']

        if yaml_start == 'random':
            start = np.array([border_x + random.random()*(self.size_x - 2*border_x),border_y + random.random()*(self.size_y - 2*border_y),border_z + random.random()*(self.size_z - 2*border_z)], dtype=float)
        elif yaml_start == 'random_low':
            start = np.array([border_x + random.random()*(self.size_x - 2*border_x),border_y + random.random()*(self.size_y - 2*border_y),0], dtype=float)
        elif yaml_start == 'left':
            start = np.array([border_x + random.random()*(self.size_x/2 - 2*border_x),border_y + random.random()*(self.size_y - 2*border_y),border_z + random.random()*(self.size_z - 2*border_z)], dtype=float)
        elif yaml_start == 'left_low':
            start = np.array([border_x + random.random()*(self.size_x/2 - 2*border_x),border_y + random.random()*(self.size_y - 2*border_y),0], dtype=float)
        else:
            start = np.array(yaml_start, dtype=float)
        return start

    def set_target(self, start):
        border_x = self.size_x/(10*self.render_ratio)
        border_y = self.size_y/(10*self.render_ratio)
        border_z = self.size_z/10

        if self.train_or_test == 'train':
            yaml_target = yaml_p['target_train']
        else:
            yaml_target = yaml_p['target_test']

        if yaml_target == 'random':
            target = np.array([border_x + random.random()*(self.size_x - 2*border_x),border_y + random.random()*(self.size_y - 2*border_y),border_z + random.random()*(self.size_z - 2*border_z)], dtype=float)
        elif yaml_target == 'random_low':
            target = np.array([border_x + random.random()*(self.size_x - 2*border_x),border_y + random.random()*(self.size_y - 2*border_y),0], dtype=float)
        elif yaml_target == 'right':
            target = np.array([start[0] + random.random()*(self.size_x - start[0] - border_x),border_y + random.random()*(self.size_y - 2*border_y),border_z + random.random()*(self.size_z - 2*border_z)], dtype=float)
        elif yaml_target == 'right_low':
            target = np.array([start[0] + random.random()*(self.size_x - start[0] - border_x),border_y + random.random()*(self.size_y - 2*border_y),0], dtype=float)
        else:
            target = np.array(yaml_target, dtype=float)

        return target

    def character_v(self, position):
        world_compressed = self.ae.compress(self.world, self.character.position, self.character.ceiling)
        character_v = character(self.size_x, self.size_z, position, self.character.target, self.T, self.world, world_compressed)
        v_x = self.world[-3][int(position[0]), int(position[1])] #approximate current velocity as velocity of world_map
        v_z = self.world[-2][int(position[0]), int(position[1])]
        if yaml_p['physics']:
            character_v.state[2:4] = [v_x, v_z]

        return character_v.state
