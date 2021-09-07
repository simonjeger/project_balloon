from human_autoencoder import HAE
from build_autoencoder import VAE
from build_character import character
from build_render import render

import pandas as pd
import matplotlib.pyplot as plt
from gym import Env, logger
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
        self.success = False
        self.success_n = 0
        self.epi_n = epi_n
        self.step_n = step_n
        self.seed = 0

        # initialize autoencoder object
        if yaml_p['autoencoder'][0:3] == 'HAE':
            self.ae = HAE()
        if yaml_p['autoencoder'] == 'VAE':
            self.ae = VAE()
            self.ae.load_weights('autoencoder/model_' + str(yaml_p['vae_nr']) + '.pt')

        # load new world to get size_x, size_z
        self.load_new_world()

        # action we can take: -1 to 1
        self.action_space = Box(low=-1.0, high=1.0, shape=(1 + int(yaml_p['memo']),), dtype=np.float64) #continuous space

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

    def step(self, action, roll_out=False):
        # Interpolate the get the current world
        self.interpolate_world(self.character.t)

        # Update compressed wind map
        self.world_compressed = self.ae.compress(self.world, self.character.position, self.character.ceiling)
        self.world_compressed /= yaml_p['unit_xy'] #so it's in simulation units and makes sense for the normalization in character.py

        coord = [int(i) for i in np.round(self.character.position)] #convert position into int so I can use it as index
        done = False

        # move character
        in_bounds = self.character.update(action, self.world, self.world_compressed, roll_out)
        done = self.cost(in_bounds)

        if not roll_out:
            # logger
            if self.writer is not None:
                if (self.step_n % yaml_p['log_frequency'] == 0) & (not done):
                    self.writer.add_scalar('epi_n', self.epi_n , self.step_n)
                    self.writer.add_scalar('position_x', self.character.position[0], self.step_n)
                    self.writer.add_scalar('position_y', self.character.position[1], self.step_n)
                    self.writer.add_scalar('position_z', self.character.position[2], self.step_n)
                    self.writer.add_scalar('reward_step', self.reward_step, self.step_n)
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

    def cost(self, in_bounds):
        init_proj_min = np.sqrt(((self.character.target[0] - self.character.start[0])*self.render_ratio/self.radius_xy)**2 + ((self.character.target[1] - self.character.start[1])*self.render_ratio/self.radius_xy)**2 + ((self.character.target[2] - self.character.start[2])/self.radius_z)**2)

        if in_bounds:
            # calculate reward
            if self.character.min_proj_dist <= 1:
                self.reward_step = yaml_p['hit']
                self.success = True
                self.success_n += 1
                done = True
            else:
                self.reward_step = yaml_p['step']*yaml_p['delta_t'] + abs(self.character.U)*yaml_p['action']
                done = False

            if self.character.t <= 0:
                self.reward_step = yaml_p['overtime'] + (init_proj_min - self.character.min_proj_dist)/init_proj_min * yaml_p['min_proj_dist']
                self.success = False
                done = True

        else:
            self.reward_step = yaml_p['bounds'] + (init_proj_min - self.character.min_proj_dist)/init_proj_min * yaml_p['min_proj_dist']
            self.character.t = 0
            self.success = False
            done = True

        self.reward_epi += self.reward_step
        self.reward_list.append(self.reward_step)
        return done

    def render(self, mode=False): #mode = False is needed so I can distinguish between when I want to render and when I don't
        self.render_machine.make_render(self.character, self.reward_step, self.reward_epi, self.world_name, self.ae.window_size, self.radius_xy, self.radius_z, self.train_or_test, self.path_roll_out)

    def reset(self, roll_out=False):
        # load new world
        if not roll_out:
            self.load_new_world()
        else:
            self.interpolate_world(yaml_p['T']) #still set back the world to time = takeoff_time

        self.reward_step = 0
        self.reward_epi = 0

        # Set problem
        if not roll_out:
            start = self.set_start()
        else:
            start = self.character.path[0]
        self.set_target(start)

        # if started "under ground"
        above_ground_start = self.size_z/100
        above_ground_target = self.size_z/100
        x = np.linspace(0,self.size_x,len(self.world[0,:,0,0]))
        y = np.linspace(0,self.size_y,len(self.world[0,0,:,0]))

        f = scipy.interpolate.interp2d(x,y,self.world[0,:,:,0].T)

        if start[2] <= f(start[0], start[1])[0] + above_ground_start:
            start[2] = f(start[0], start[1])[0] + above_ground_start

        if self.target[2] <= f(self.target[0], self.target[1])[0] + above_ground_target:
            self.target[2] = f(self.target[0], self.target[1])[0] + above_ground_target

        # Initial compressed wind map
        self.world_compressed = self.ae.compress(self.world, start, self.size_z)

        if yaml_p['environment'] == 'python3':
            self.character = character(self.size_x, self.size_y, self.size_z, start, self.target, self.radius_xy, self.radius_z, self.T, self.world, self.world_compressed, self.train_or_test, self.seed)

            # avoid impossible szenarios
            min_space = self.size_z*yaml_p['min_space']
            pos_x = int(np.clip(self.character.position[0],0,self.size_x - 1))
            pos_y = int(np.clip(self.character.position[1],0,self.size_y - 1))
            if (self.character.ceiling - self.world[0,pos_x,pos_y,0] < min_space) | (np.mean(self.character.ceiling - self.world[0,:,:,0]) < min_space):
                self.reset(roll_out=roll_out)
            self.reward_list = []

        elif yaml_p['environment'] == 'xplane':
            self.character = character_xplane(self.size_x, self.size_y, self.size_z, start, self.target, self.radius_xy, self.radius_z, self.T, self.world, self.world_compressed, self.train_or_test, self.seed)

        """
        # generate world_squished for rendering later
        from preprocess_wind import preprocess
        if yaml_p['render'] & (not roll_out):
            preprocess(self.world, self.character.ceiling)
        """

        return self.character.state

    def load_new_world(self):
        # choose random world_map
        if self.train_or_test == 'test':
            np.random.seed(self.seed)
            self.seed +=1
        self.world_name = np.random.choice(os.listdir(yaml_p['data_path'] + self.train_or_test + '/tensor'))

        # remove suffix
        length = len(self.world_name)
        self.world_name = self.world_name[:length - 3]

        # Interpolate the get the current world
        if self.train_or_test == 'test':
            np.random.seed(self.seed)
            self.seed +=1
        self.takeoff_time = np.random.randint(0,23*60*60 - yaml_p['T'])

        self.interpolate_world(yaml_p['T'])

        # define world size
        self.size_x = len(self.world[-1,:,:])
        self.size_y = len(self.world[-1,:,:][0])
        self.size_z = len(self.world[-1,:,:][0][0])

    def interpolate_world(self,t):
        tss = yaml_p['T'] - t + self.takeoff_time

        h = int(tss/60/60)
        s = tss - h*60*60
        p = s/60/60

        if h < 23:
            i = 1
        else:
            i = 0

        self.world_0 = torch.load(yaml_p['data_path'] + self.train_or_test + '/tensor_t/' + self.world_name + '_'  + str(h).zfill(2) + '.pt')
        self.world_1 = torch.load(yaml_p['data_path'] + self.train_or_test + '/tensor_t/' + self.world_name + '_'  + str(h+i).zfill(2) + '.pt')

        self.world = p*(self.world_1 - self.world_0) + self.world_0
        torch.save(self.world, yaml_p['data_path'] + self.train_or_test + '/tensor/' + self.world_name + '.pt')


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
            if self.train_or_test == 'test':
                np.random.seed(self.seed)
                self.seed +=1
            start = np.floor(start) + np.append(np.random.uniform(0,1,2),[0]) #randomize start position without touching the z-axis
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
            self.target = np.array([border_x + random.random()*(self.size_x - 2*border_x),border_y + random.random()*(self.size_y - 2*border_y),border_z + random.random()*(self.size_z - 2*border_z)], dtype=float)
        elif yaml_target == 'random_low':
            self.target = np.array([border_x + random.random()*(self.size_x - 2*border_x),border_y + random.random()*(self.size_y - 2*border_y),0], dtype=float)
        elif yaml_target == 'right':
            self.target = np.array([start[0] + random.random()*(self.size_x - start[0] - border_x),border_y + random.random()*(self.size_y - 2*border_y),border_z + random.random()*(self.size_z - 2*border_z)], dtype=float)
        elif yaml_target == 'right_low':
            self.target = np.array([start[0] + random.random()*(self.size_x - start[0] - border_x),border_y + random.random()*(self.size_y - 2*border_y),0], dtype=float)
        else:
            self.target = np.array(yaml_target, dtype=float)
