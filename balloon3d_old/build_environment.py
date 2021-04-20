
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

class balloon3d(Env):
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
        regular_state_space_low = np.array([0,0,0,0,0,0,0,0,0]) #residual to target, distance to border
        wind_compressed_state_space_low = np.array([-1]*self.ae.bottleneck)
        regular_state_space_high = np.array([self.size_x,self.size_y,self.size_z,0,self.size_x,0,self.size_y,0,self.size_z])
        wind_compressed_state_space_high = np.array([1]*self.ae.bottleneck)
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
        # Update compressed wind map
        window = self.ae.window(self.wind_map, self.character.position)
        window = np.array([window])
        self.wind_compressed = self.ae.compress(window)

        coord = [int(i) for i in np.round(self.character.position)] #convert position into int so I can use it as index
        done = False

        # move character
        in_bounds = self.character.update(action, self.wind_map, self.wind_compressed)
        done = self.cost(in_bounds)

        # set placeholder for info
        info = {}

        # return step information
        return self.character.state, self.reward_step, done, info


    def cost(self, in_bounds):
        if in_bounds:
            # calculate reward
            distance = np.sqrt(self.character.residual[0]**2 + self.character.residual[1]**2 + self.character.residual[2]**2)
            radius = 1
            ramp = 30
            if distance <= radius:
                self.reward_step = self.character.t
                done = True
            else:
                self.reward_step = max((ramp - distance)/ramp, - 1/self.T)
                done = False

            if self.character.t <= 0:
                self.reward_step = 0
                done = True

        else:
            self.reward_step = -1
            self.character.t = 0
            done = True

        self.reward_epi += self.reward_step
        return done

    def render(self, mode=False): #mode = False is needed so I can distinguish between when I want to render and when I don't
        build_render(self.character, self.reward_step, self.reward_epi, self.ae.window_size, self.train_or_test)

    def reset(self):
        # load new world
        self.load_new_world()
        self.reward_step = 0
        self.reward_epi = 0

        # Set problem
        border = 2
        start = np.array([self.size_x/2,self.size_y/2,0], dtype=float)
        target = np.array([random.randint(border, self.size_x - border),random.randint(border, self.size_y - border),random.randint(border, self.size_z - border)], dtype=float)

        # Initial compressed wind map
        window = self.ae.window(self.wind_map, start)
        window = np.array([window])
        self.wind_compressed = self.ae.compress(window)

        self.character = character(self.size_x, self.size_y, self.size_z, start, target, self.T, self.wind_map, self.wind_compressed) #weight should be 1000 kg for realistic dimensions
        return self.character.state

    def load_new_world(self):
        # choose random wind_map
        world_name = random.choice(os.listdir('data/' + self.train_or_test + '/tensor'))
        # remove suffix
        length = len(world_name)
        world_name = world_name[:length - 3]
        # read in wind_map
        self.wind_map = torch.load('data/' + self.train_or_test + '/tensor/' + world_name + '.pt')

        # define world size
        self.size_x = len(self.wind_map)
        self.size_y = len(self.wind_map[0])
        self.size_z = len(self.wind_map[0][0])
