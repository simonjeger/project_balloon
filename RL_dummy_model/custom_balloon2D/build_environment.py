
from build_render import build_render
from build_autoencoder import Autoencoder
from fake_autoencoder import fake_Autoencoder
from build_character import character

from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import torch
from random import gauss
import os

class balloon2d(Env):
    def __init__(self, train_or_test):
        # which data to use
        self.train_or_test = train_or_test

        # initialize autoencoder object
        self.ae = Autoencoder()
        self.ae.autoencoder_model.load_weights('weights_autoencoder/ae_weights.h5f')
        #self.ae = fake_Autoencoder()

        # load new world to get size_x, size_z
        self.load_new_world()

        # action we can take: down, stay, up
        self.action_space = Discrete(3) #discrete space

        # location array in x and z
        regular_state_space_low = np.array([0,0,0,0,0,0]) #residual to target, distance to border
        wind_compressed_state_space_low = np.array([-1]*len(self.wind_compressed))
        regular_state_space_high = np.array([self.size_x,self.size_z,0,self.size_x,0,self.size_z])
        wind_compressed_state_space_high = np.array([1]*len(self.wind_compressed))
        self.observation_space = Box(low=np.concatenate((regular_state_space_low, wind_compressed_state_space_low), axis=0), high=np.concatenate((regular_state_space_high, wind_compressed_state_space_high), axis=0)) #ballon_x = [0,...,100], balloon_z = [0,...,30], error_x = [0,...,100], error_z = [0,...,30]

        # set start position
        self.start = np.array([10,15], dtype=float)
        # set target position
        self.target = np.array([90,20], dtype=float)
        # set maximal duration of flight
        self.T = 200
        # initialize state and time

        self.reset()

    def step(self, action):
        coord = [int(i) for i in np.round(self.character.position)] #convert position into int so I can use it as index
        done = False

        # move character
        in_bounds = self.character.update(action, self.wind_map, self.wind_compressed)
        if in_bounds:
            # calculate reward
            distance = np.sqrt(self.character.residual[0]**2 + self.character.residual[1]**2)
            radius = 1
            ramp = 30
            if distance <= radius:
                self.reward = np.sqrt(self.character.t)
                done = True
            else:
                self.reward = max(ramp - distance, 0)/ramp

            if self.character.t <= 0:
                self.reward = -1
                done = True

        else:
            self.reward = 0
            self.character.t = -np.sqrt(self.character.t)
            done = True

        # set placeholder for info
        info = {}

        # return step information
        return self.character.state, self.reward, done, info

    def render(self, mode=False): #mode = False is needed so I can distinguish between when I want to render and when I don't
        build_render(self.character, self.reward, self.world_name, self.ae.window_size, self.train_or_test)

    def reset(self):
        # load new world
        self.load_new_world()

        border = 2
        start = np.array([self.size_x/2,0], dtype=float)
        target = np.array([random.randint(border, self.size_x - border),random.randint(border, self.size_z - border)], dtype=float)
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

        # just to initialize len() of variable
        x_train_window = self.ae.window(self.wind_map)
        x_train_window = np.array([x_train_window])
        self.wind_compressed = self.ae.compress(x_train_window) # just to initialize len() of variable

        self.mean_x = self.wind_map[:,:,0]
        self.mean_z = self.wind_map[:,:,1]
        self.sig_xz = self.wind_map[:,:,2]

        # define world size
        self.size_x = len(self.mean_x)
        self.size_z = len(self.mean_x[0])
