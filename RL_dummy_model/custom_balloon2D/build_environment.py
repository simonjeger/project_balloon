from build_render import build_render

import matplotlib.pyplot as plt
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

        # load new world to get size_x, size_z
        self.load_new_world()

        # action we can take: down, stay, up
        self.action_space = Discrete(3) #discrete space
        # location array in x and z
        regular_state_space_low = np.array([0,0,0,0])
        wind_compressed_state_space_low = np.array([-1]*len(self.wind_compressed))
        regular_state_space_high = np.array([self.size_x,self.size_z,self.size_x,self.size_z])
        wind_compressed_state_space_high = np.array([1]*len(self.wind_compressed))
        self.observation_space = Box(low=np.concatenate((regular_state_space_low, wind_compressed_state_space_low), axis=0), high=np.concatenate((regular_state_space_high, wind_compressed_state_space_high), axis=0)) #ballon_x = [0,...,100], balloon_z = [0,...,30], error_x = [0,...,100], error_z = [0,...,30]
        # set start position
        self.start = np.array([10,15], dtype=float)
        # set target position
        self.target = np.array([90,20], dtype=float)
        # set target radius
        self.radius = 5
        # set maximal duration of flight
        self.T = 1000
        # initialize state and time
        self.reset()

    def step(self, action):
        coord = [int(i) for i in np.round(self.state)] #convert position into int so I can use it as index
        in_bounds = (0 <= coord[0] < self.size_x) & (0 <= coord[1] < self.size_z) #if still within bounds
        done = False

        if in_bounds:
            # apply wind
            self.state[0] += self.mean_x[coord[0]][coord[1]]
            self.state[1] += self.mean_z[coord[0]][coord[1]]

            """
            step_dist = np.sqrt(self.mean_x[coord[0]][coord[1]]**2 + self.mean_z[coord[0]][coord[1]]**2)
            step_t = 1

            while step_t > 0:
                speed_x = self.mean_x[coord[0]][coord[1]]
                speed_z = self.mean_z[coord[0]][coord[1]]

                sign_x = np.sign(speed_x)
                sign_z = np.sign(speed_z)

                if sign_x == 1:
                    dist_x = np.ceil(self.state[0]) - self.state[0]
                else:
                    dist_x = self.state[0] - np.floor(self.state[0])
                if sign_z == 1:
                    dist_z = np.ceil(self.state[1]) - self.state[1]
                else:
                    dist_z = self.state[1] - np.floor(self.state[1])

                time_to_border_x = dist_x / abs(speed_x)
                time_to_border_z = dist_z / abs(speed_z)
                time_in_this_box = min(min(time_to_border_x, time_to_border_z),1)

                self.state[0] += speed_x * time_in_this_box
                self.state[1] += speed_z * time_in_this_box

                step_t -= time_in_this_box

                print('state: ' + str(self.state))
                print('dist_x: ' + str(dist_x))
                print('dist_z: ' + str(dist_z))
                print('speed_x: ' + str(speed_x))
                print('speed_z: ' + str(speed_z))
                print('time_to_border_x: ' + str(time_to_border_x))
                print('time_to_border_z: ' + str(time_to_border_z))
                print('time_in_this_box: ' + str(time_in_this_box))
                print('step_t :' + str(step_t))
                print('---------------')
            """

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
            worst = np.sqrt(self.size_x**2 + self.size_z**2)*self.t
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
        build_render(self.state, self.target, self.size_x, self.size_z, self.t, self.world_name, self.train_or_test)

    def reset(self):
        # load new world
        self.load_new_world()

        # reset initial position
        self.start = np.array([random.randint(0, self.size_x),random.randint(0, self.size_z)], dtype=float)
        self.target = np.array([random.randint(0, self.size_x),random.randint(0, self.size_z)], dtype=float)
        self.state = np.concatenate((self.start.flatten(), self.target.flatten(), self.wind_compressed.flatten()), axis=0)

        # reset flight time
        self.t = self.T
        return self.state

    def load_new_world(self):
        # choose random wind_map
        self.world_name = random.choice(os.listdir('data/' + self.train_or_test + '/tensor'))
        # remove suffix
        length = len(self.world_name)
        self.world_name = self.world_name[:length - 3]
        # read in wind_map
        wind_map = torch.load('data/' + self.train_or_test + '/tensor/' + self.world_name + '.pt')
        self.wind_compressed = torch.load('data/' + self.train_or_test + '/tensor_comp/' + self.world_name + '.pt')

        """
        self.mean_x = wind_map[:,:,0]
        self.var_x = wind_map[:,:,1] + wind_map[:,:,2]
        self.mean_z = wind_map[:,:,3]
        self.var_z = wind_map[:,:,4] + wind_map[:,:,5]
        """
        
        self.mean_x = wind_map[:,:,0]
        self.mean_z = wind_map[:,:,1]

        # define world size
        self.size_x = len(self.mean_x)
        self.size_z = len(self.mean_x[0])
