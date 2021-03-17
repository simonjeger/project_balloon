
from build_render import build_render
from build_autoencoder import Autoencoder
from fake_autoencoder import fake_Autoencoder
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
    def __init__(self, train_or_test):
        # which data to use
        self.train_or_test = train_or_test

        # initialize autoencoder object
        self.ae = Autoencoder()
        self.ae.autoencoder_model.load_weights('process' + str(yaml_p['process_nr']).zfill(5) + '/weights_autoencoder/ae_weights.h5f')
        #self.ae = fake_Autoencoder()

        # load new world to get size_x, size_z
        self.load_new_world()

        # action we can take: down, stay, up
        self.action_space = Discrete(3) #discrete space

        # set maximal duration of flight
        self.T = yaml_p['T']

        # location array in x and z
        regular_state_space_low = np.array([0,0,0,0,0,0,0]) #residual to target, distance to border, time
        wind_compressed_state_space_low = np.array([-1]*self.ae.bottleneck)
        regular_state_space_high = np.array([self.size_x,self.size_z,0,self.size_x,0,self.size_z,self.T])
        wind_compressed_state_space_high = np.array([1]*self.ae.bottleneck)
        self.observation_space = Box(low=np.concatenate((regular_state_space_low, wind_compressed_state_space_low), axis=0), high=np.concatenate((regular_state_space_high, wind_compressed_state_space_high), axis=0)) #ballon_x = [0,...,100], balloon_z = [0,...,30], error_x = [0,...,100], error_z = [0,...,30]

        # initialize state and time
        self.reset()

        # delete old path file if it exists
        if os.path.isfile('process' + str(yaml_p['process_nr']).zfill(5) + '/log_environment.csv'):
            os.remove('process' + str(yaml_p['process_nr']).zfill(5) + '/log_environment.csv')
        self.epi = 0

    def step(self, action):
        # Update compressed wind map
        window = self.ae.window(self.wind_map, self.character.position[0])
        window = np.array([window])
        self.wind_compressed = self.ae.compress(window)

        coord = [int(i) for i in np.round(self.character.position)] #convert position into int so I can use it as index
        done = False

        # move character
        in_bounds = self.character.update(action, self.wind_map, self.wind_compressed)
        done = self.cost(in_bounds)

        # write in log file
        df = pd.DataFrame([[self.epi, self.size_x, self.size_z, self.character.position[0], self.character.position[1], self.character.target[0], self.character.target[1], self.reward_step, self.reward_epi]])
        df.to_csv('process' + str(yaml_p['process_nr']).zfill(5) + '/log_environment.csv', mode='a', header=False, index=False)
        if done:
            self.epi += 1

        # set placeholder for info
        info = {}

        # return step information
        return self.character.state, self.reward_step, done, info

    def cost(self, in_bounds):
        if in_bounds:
            # calculate reward
            radius = yaml_p['radius']
            ramp = yaml_p['ramp']
            exp = yaml_p['exp']
            lam = yaml_p['lam']
            distance = np.sqrt(self.character.residual[0]**2 + self.character.residual[1]**2)

            if distance <= radius:
                self.reward_step = self.character.t
                done = True
            else:
                self.reward_step = max(((ramp-distance)/ramp), 0)**exp -1/self.T + abs(self.character.action - 1)*lam
                done = False

            if self.character.t <= 0:
                self.reward_step = yaml_p['overtime']
                done = True

        else:
            self.reward_step = -self.T + (self.T - self.character.t)/self.T #hitting a wall should always cost the same
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
        border = 2
        if isinstance(yaml_p['start'], str):
            start = np.array([random.randint(border, self.size_x/2 - border),0], dtype=float)
        else:
            start = np.array(yaml_p['start'], dtype=float)
        if isinstance(yaml_p['target'], str):
            #target = np.array([random.randint(border + self.size_x/2, self.size_x - border),random.randint(border, self.size_z - border)], dtype=float)
            target = np.array([25,random.randint(border, self.size_z - border)], dtype=float)
        else:
            target = np.array(yaml_p['target'], dtype=float)

        # Initial compressed wind map
        window = self.ae.window(self.wind_map, start[0])
        window = np.array([window])
        self.wind_compressed = self.ae.compress(window)

        self.character = character(self.size_x, self.size_z, start, target, self.T, self.wind_compressed) #weight should be 1000 kg for realistic dimensions
        return self.character.state

    def load_new_world(self):
        # choose random wind_map
        self.world_name = random.choice(os.listdir('data/' + self.train_or_test + '/tensor'))
        # remove suffix
        length = len(self.world_name)
        self.world_name = self.world_name[:length - 3]
        # read in wind_map
        self.wind_map = torch.load('data/' + self.train_or_test + '/tensor/' + self.world_name + '.pt')

        self.mean_x = self.wind_map[:,:,0]
        self.mean_z = self.wind_map[:,:,1]
        self.sig_xz = self.wind_map[:,:,2]

        # define world size
        self.size_x = len(self.mean_x)
        self.size_z = len(self.mean_x[0])
