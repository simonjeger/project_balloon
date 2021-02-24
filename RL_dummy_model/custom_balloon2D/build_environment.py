from build_render import build_render

from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import torch
from random import gauss
import os

class balloon2d(Env):
    def __init__(self):
        # load new world to get size_x, size_z
        self.load_new_world()

        # action we can take: down, stay, up
        self.action_space = Discrete(3) #discrete space
        # location array in x and z
        self.observation_space = Box(low=np.array([0,0,0,0]), high=np.array([self.size_x,self.size_z,self.size_x,self.size_z])) #ballon_x = [0,...,100], balloon_z = [0,...,30], error_x = [0,...,100], error_z = [0,...,30]
        # set start position
        self.start = np.array([10,15])
        # set target position
        self.target = np.array([90,20])
        # set target radius
        self.radius = 5
        # set maximal duration of flight
        self.T = 200
        # initialize state and time
        self.reset()

    def step(self, action):
        coord = [int(i) for i in np.round(self.state)] #convert position into int so I can use it as index
        in_bounds = (0 <= coord[0] < self.size_x) & (0 <= coord[1] < self.size_z) #if still within bounds
        done = False

        if in_bounds:
            # apply wind
            self.state[0] += gauss(self.mean_x[coord[0]][coord[1]],self.var_x[coord[0]][coord[1]])
            self.state[1] += gauss(self.mean_z[coord[0]][coord[1]],self.var_z[coord[0]][coord[1]])
            # apply action
            self.state[1] += action -1
            # reduce flight length by 1 second
            self.t -= 1
            # calculate reward
            distance = np.abs(self.state[0] - self.target[0]) + np.abs(self.state[1] - self.target[1])
            reward = - distance

            # check if flight is done
            if (self.t <= 0) | (distance <= self.radius):
                done = True
        else:
            worst = np.sqrt(self.size_x**2 + self.size_z**2) * self.t
            reward = - worst
            self.t = 0
            done = True

        # apply noise
        #self.state[0] += random.randint(-1,1)
        # set placeholder for info
        info = {}

        # return step information
        return self.state, reward, done, info

    def render(self, mode=False): #mode = False is needed so I can distinguish between when I want to render and when I don't
        build_render(self.state, self.target, self.size_x, self.size_z, self.t)

    def reset(self):
        # load new world
        self.load_new_world()

        # reset initial position
        offset = self.start - self.target
        self.state = np.concatenate((self.start.flatten(), offset.flatten()), axis=0)
        # reset flight time
        self.t = self.T

        return self.state

    def load_new_world(self):
        # choose random wind_map
        filename = random.choice(os.listdir('data/tensor'))
        # read in wind_map
        wind_map = torch.load('data/tensor/' + filename)
        self.mean_x = wind_map[0]
        self.var_x = wind_map[1] + wind_map[2]
        self.mean_z = wind_map[3]
        self.var_z = wind_map[4] + wind_map[5]
        # define world size
        self.size_x = len(self.mean_x)
        self.size_z = len(self.mean_x[0])
