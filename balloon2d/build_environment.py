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
    def __init__(self, train_or_test, writer=None):
        # which data to use
        self.train_or_test = train_or_test
        self.writer = writer

        # initialize autoencoder object
        if yaml_p['autoencoder'] == 'HAE':
            self.ae = HAE()
        if yaml_p['autoencoder'] == 'VAE':
            self.ae = VAE()

        # load new world to get size_x, size_z
        self.load_new_world()

        # action we can take: down, stay, up
        self.action_space = Discrete(3) #discrete space

        # set maximal duration of flight
        self.T = yaml_p['T']

        # location array in x and z
        if yaml_p['physics']:
            regular_state_space_low = np.array([-self.size_x,-self.size_z,-np.inf,-np.inf,0,0,0,0]) #residual to target, velocity, distance to border
            regular_state_space_high = np.array([self.size_x,self.size_z,np.inf,np.inf,self.size_x,self.size_x,self.size_z,self.size_z])
        else:
            regular_state_space_low = np.array([-self.size_x,-self.size_z,0,0,0,0]) #residual to target, velocity, distance to border
            regular_state_space_high = np.array([self.size_x,self.size_z,self.size_x,self.size_x,self.size_z,self.size_z])
        world_compressed_state_space_low = np.array([-1]*self.ae.bottleneck)
        world_compressed_state_space_high = np.array([1]*self.ae.bottleneck)
        self.observation_space = Box(low=np.concatenate((regular_state_space_low, world_compressed_state_space_low), axis=0), high=np.concatenate((regular_state_space_high, world_compressed_state_space_high), axis=0)) #ballon_x = [0,...,100], balloon_z = [0,...,30], error_x = [0,...,100], error_z = [0,...,30]

        # initialize state and time
        self.reset()

        self.step_n = 0
        self.epi_n = 0

    def step(self, action):
        # Update compressed wind map
        self.world_compressed = self.ae.compress(self.world, self.character.position)

        coord = [int(i) for i in np.round(self.character.position)] #convert position into int so I can use it as index
        done = False

        # move character
        in_bounds = self.character.update(action, self.world_compressed)
        done = self.cost(in_bounds)

        # logger
        if self.writer is not None:
            if (self.step_n % yaml_p['log_frequency'] == 0) & (not done):
                self.writer.add_scalar('episode', self.epi_n , self.step_n)
                self.writer.add_scalar('position_x', self.character.position[0], self.step_n)
                self.writer.add_scalar('position_z', self.character.position[1], self.step_n)
                self.writer.add_scalar('reward_step', self.reward_step, self.step_n)
            if done:
                self.writer.add_scalar('episode', self.epi_n , self.step_n)
                self.writer.add_scalar('position_x', self.character.position[0], self.step_n)
                self.writer.add_scalar('position_z', self.character.position[1], self.step_n)
                self.writer.add_scalar('reward_step', self.reward_step, self.step_n)

                self.writer.add_scalar('size_x', self.size_x , self.step_n)
                self.writer.add_scalar('size_z', self.size_z , self.step_n)
                self.writer.add_scalar('target_x', self.character.target[0], self.step_n)
                self.writer.add_scalar('target_z', self.character.target[1], self.step_n)
                self.writer.add_scalar('reward_epi', self.reward_epi, self.step_n)

        self.step_n += 1
        if done:
            self.epi_n += 1

        # set placeholder for info
        info = {}

        # return step information
        return self.character.state, self.reward_step, done, info

    def cost(self, in_bounds):
        init_min = np.sqrt((self.character.target[0] - self.character.start[0])**2 + (self.character.target[1] - self.character.start[1])**2)
        if in_bounds:
            # calculate reward
            distance = np.sqrt(self.character.residual[0]**2 + self.character.residual[1]**2)

            if distance <= yaml_p['radius']:
                self.reward_step = yaml_p['hit']
                done = True
            else:
                self.reward_step = yaml_p['step'] + abs(self.character.action - 1)*yaml_p['action']
                done = False

            if self.character.t <= 0:
                self.reward_step = yaml_p['overtime'] + (init_min - self.character.min_distance)/init_min * yaml_p['min_distance']
                done = True

        else:
            self.reward_step = yaml_p['bounds'] + (init_min - self.character.min_distance)/init_min * yaml_p['min_distance']
            self.character.t = 0
            done = True

        self.reward_epi += self.reward_step
        return done

    def render(self, mode=False): #mode = False is needed so I can distinguish between when I want to render and when I don't
        build_render(self.character, self.reward_step, self.reward_epi, self.world_name, self.ae.window_size, self.train_or_test)

    def reset(self):
        # load new world
        self.load_new_world()
        self.reward_step = 0
        self.reward_epi = 0

        # Set problem
        border_x = self.size_x/10
        border_z = self.size_z/10
        above_ground = self.size_z/5

        if yaml_p['start'] == 'random':
            start = np.array([border_x + random.random()*(self.size_x - 2*border_x),border_z + random.random()*(self.size_z - 2*border_z)], dtype=float)
        elif yaml_p['start'] == 'random_low':
            start = np.array([border_x + random.random()*(self.size_x - 2*border_x),0], dtype=float)
        elif yaml_p['start'] == 'left':
            start = np.array([border_x + random.random()*(self.size_x/2 - 2*border_x),border_z + random.random()*(self.size_z - 2*border_z)], dtype=float)
        elif yaml_p['start'] == 'left_low':
            start = np.array([border_x + random.random()*(self.size_x/2 - 2*border_x),0], dtype=float)
        else:
            start = np.array(yaml_p['start'], dtype=float)

        if yaml_p['target'] == 'random':
            target = np.array([border_x + random.random()*(self.size_x - 2*border_x),border_z + random.random()*(self.size_z - 2*border_z)], dtype=float)
        elif yaml_p['target'] == 'random_low':
            target = np.array([border_x + random.random()*(self.size_x - 2*border_x),0], dtype=float)
        elif yaml_p['target'] == 'right':
            target = np.array([start[0] + random.random()*(self.size_x - start[0] - border_x),border_z + random.random()*(self.size_z - 2*border_z)], dtype=float)
        elif yaml_p['target'] == 'right_low':
            target = np.array([start[0] + random.random()*(self.size_x - start[0] - border_x),0], dtype=float)
        else:
            target = np.array(yaml_p['target'], dtype=float)

        # if started "under ground"
        x = np.linspace(0,self.size_x,len(self.world[0,:,0]))

        if start[1] <= np.interp(start[0],x,self.world[0,:,0]):
            start[1] = np.ceil(np.interp(start[0],x,self.world[0,:,0]))

        if target[1] <= np.interp(target[0],x,self.world[0,:,0]) + above_ground:
            target[1] = np.interp(target[0],x,self.world[0,:,0]) + above_ground

        # Initial compressed wind map
        self.world_compressed = self.ae.compress(self.world, start)

        self.character = character(self.size_x, self.size_z, start, target, self.T, self.world, self.world_compressed)

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
        self.size_z = len(self.world[-1,:,:][0])

    def character_v(self, position):
        world_compressed = self.ae.compress(self.world, self.character.position)
        character_v = character(self.size_x, self.size_z, position, self.character.target, self.T, self.world, world_compressed)
        v_x = self.world[-3][int(position[0]), int(position[1])] #approximate current velocity as velocity of world_map
        v_z = self.world[-2][int(position[0]), int(position[1])]
        if yaml_p['physics']:
            character_v.state[2:4] = [v_x, v_z]

        return character_v.state
