import torch
import numpy as np
from collections import deque
import os
import pandas as pd
import pfrl
import copy
from pathlib import Path
import shutil
from distutils.dir_util import copy_tree

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

class QFunction(torch.nn.Module):

    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.l1 = torch.nn.Linear(obs_size, 50)
        self.l2 = torch.nn.Linear(50, 50)
        self.l3 = torch.nn.Linear(50, n_actions)

    def forward(self, x):
        h = x
        h = torch.nn.functional.relu(self.l1(h))
        h = torch.nn.functional.relu(self.l2(h))
        h = self.l3(h)

        return pfrl.action_value.DiscreteActionValue(h)

class Agent:
    def __init__(self, env, writer=None):
        self.env = env
        self.stash = [0]*yaml_p['phase']
        self.writer = writer

        acts = env.action_space
        obs = env.observation_space
        self.qfunction = QFunction(obs.shape[0],acts.n)
        #self.qfunction.apply(self.weights_init) # delibratly initialize weights of NN as defined in function below

        epsi_high = yaml_p['epsi_high']
        epsi_low = yaml_p['epsi_low']
        decay = yaml_p['decay']
        scale = yaml_p['scale']

        if yaml_p['explorer_type'] == 'LinearDecayEpsilonGreedy':
            explorer = pfrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=epsi_high, end_epsilon=epsi_low, decay_steps=decay, random_action_func=env.action_space.sample)
        elif yaml_p['explorer_type'] == 'Boltzmann':
            explorer = pfrl.explorers.Boltzmann()
        elif yaml_p['explorer_type'] == 'AdditiveGaussian':
            explorer = pfrl.explorers.AdditiveGaussian(scale, low=0, high=2)

        self.epi_n_update_interval = yaml_p['epi_update_interval']
        self.epi_n = 0
        self.step_n = 0
        epi_target_update_interval = yaml_p['epi_target_update_interval']

        if yaml_p['agent_type'] == 'DoubleDQN':
            self.agent = pfrl.agents.DoubleDQN(
                self.qfunction,
                torch.optim.Adam(self.qfunction.parameters(),lr=yaml_p['lr']), #in my case ADAMS
                pfrl.replay_buffers.ReplayBuffer(capacity=yaml_p['buffer_size']), #number of experiences I train my NN with
                yaml_p['gamma'], #discount factor
                explorer, #how to choose next action
                clip_delta=True,
                max_grad_norm=yaml_p['max_grad_norm'],
                replay_start_size=yaml_p['replay_start_size'], #number of experiences in replay buffer when training begins
                update_interval=yaml_p['T'], #in later parts of the code I set the timer to this, so it updates every episode
                target_update_interval=yaml_p['T'],
                minibatch_size=yaml_p['minibatch_size'], #minibatch_size used for training the q-function network
                n_times_update=yaml_p['n_times_update'], #how many times we update the NN with a new batch per update step
                phi=lambda x: x.astype(np.float32, copy=False), #feature extractor applied to observations
                gpu=-1, #actual GPU used for computation
            )

        else:
            print('please choose one of the implemented agents')

    def run_epoch(self, render):

        obs = self.env.reset()
        sum_r = 0

        while True:
            action = self.agent.act(obs) #uses self.agent.model to decide next step
            obs, reward, done, _ = self.env.step(action)

            sum_r = sum_r + reward
            self.agent.observe(obs, reward, done, False) #False is b.c. termination via time is handeled by environment

            self.step_n += 1

            if render:
                self.env.render(mode=True)

            if done:
                if self.epi_n%self.epi_n_update_interval==0:
                    self.agent.replay_updater.update_if_necessary(yaml_p['T'])

                # logger
                if self.writer is not None:
                    self.writer.add_scalar('epsilon', self.agent.explorer.epsilon , self.step_n-1) # because we do above self.step_n += 1
                self.epi_n += 1
                break

        return sum_r

    def weights_init(self, m):
        if isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, 0.0001)

    def stash_weights(self):
        path_temp = yaml_p['process_path'] + 'process' +  str(yaml_p['process_nr']).zfill(5) + '/temp_w/'
        Path(path_temp).mkdir(parents=True, exist_ok=True)
        self.agent.save(path_temp + 'temp_agent_' + str(self.epi_n%yaml_p['phase']))

    def clear_stash(self):
        dirpath = Path(yaml_p['process_path'] + 'process' +  str(yaml_p['process_nr']).zfill(5) + '/temp_w/')
        if dirpath.exists() and dirpath.is_dir():
            shutil.rmtree(dirpath)

    def save_weights(self, phase, path):
        path_temp= yaml_p['process_path'] + 'process' +  str(yaml_p['process_nr']).zfill(5) + '/temp_w/'
        name_list = os.listdir(path_temp)
        name_list.sort()

        i = np.argmax(phase)
        copy_tree(path_temp + name_list[i], path + 'weights_agent')
        print('weights saved')

    def load_weights(self, path):
        self.agent.load(path + 'weights_agent')
        print('weights loaded')

    def visualize_q_map(self):
        res = 1

        Q_vis = np.zeros((self.env.size_x*res, self.env.size_z*res,8))
        for i in range(self.env.size_x*res):
            for j in range(self.env.size_z*res):
                position = np.array([int(i/res),int(j/res)])
                obs = self.env.character_v(position)
                state = torch.Tensor(obs).unsqueeze(0)
                Q = self.agent.model.forward(state)
                Q_tar = self.agent.target_model.forward(state)

                # model
                Q_vis[i,j,0] = Q.q_values[0][0]
                Q_vis[i,j,1] = Q.q_values[0][1]
                Q_vis[i,j,2] = Q.q_values[0][2]
                Q_vis[i,j,3] = Q.greedy_actions

                # target model
                Q_vis[i,j,4] = Q_tar.q_values[0][0]
                Q_vis[i,j,5] = Q_tar.q_values[0][1]
                Q_vis[i,j,6] = Q_tar.q_values[0][2]
                Q_vis[i,j,7] = Q_tar.greedy_actions

        return Q_vis
