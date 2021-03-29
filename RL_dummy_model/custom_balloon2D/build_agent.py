import torch
import numpy as np
from collections import deque
import os
import pandas as pd
import pfrl
import copy

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
    def __init__(self, env):
        self.env = env
        self.stash = [0]*yaml_p['phase']

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

        self.epi_update_interval = yaml_p['epi_update_interval']
        self.epi = 0
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
                phi=lambda x: x.astype(np.float32, copy=False), #feature extractor applied to observations
                gpu=-1, #actual GPU used for computation
            )

        else:
            print('please choose one of the available agents')

        # initialize log file
        if os.path.isfile('process' + str(yaml_p['process_nr']).zfill(5) + '/log_agent.csv'):
            os.remove('process' + str(yaml_p['process_nr']).zfill(5) + '/log_agent.csv')

    def run_epoch(self, render):
        obs = self.env.reset()
        sum_r = 0

        while True:
            action = self.agent.act(obs) #uses self.agent.model to decide next step
            obs, reward, done, _ = self.env.step(action)

            sum_r = sum_r + reward
            self.agent.observe(obs, reward, done, False) #False is b.c. termination via time is handeled by environment

            if yaml_p['explorer_type'] == 'LinearDecayEpsilonGreedy':
                df = pd.DataFrame([[self.agent.explorer.epsilon]])
            else:
                df = pd.DataFrame([[0]])

            df.to_csv('process' + str(yaml_p['process_nr']).zfill(5) + '/log_agent.csv', mode='a', header=False, index=False)

            if render:
                self.env.render(mode=True)

            if done:
                if self.epi%self.epi_update_interval==0:
                    self.agent.replay_updater.update_if_necessary(yaml_p['T'])
                self.epi += 1
                break

        return sum_r

    def weights_init(self, m):
        if isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, 0.0001)

    def stash_weights(self):
        self.stash.pop(0)
        self.stash.append(copy.deepcopy(self.agent))

    def save_weights(self, phase, path):
        i = np.argmax(phase)
        self.stash[i].save(path + 'weights_agent')
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

    def run_fake_epoch(self, j,render):
        obs = self.env.reset()
        sum_r = 0

        if j > 300:
            j-=300

        if j > 600:
            j-=600

        if j > 900:
            j-=900

        if j > 1200:
            j-=1200

        T = yaml_p['T']

        if j<150:
            list_0 = [2]*1
            list_1 = [1]*j
            list_2 = [2]*(T - j)
            list_3 = []
        elif j<250:
            j-=150
            list_0 = [2]*1
            list_1 = [2]*j
            list_2 = [0]*j
            list_3 = [1]*(T - j)
        else:
            j-=250
            list_0 = [0]*T
            list_1 = []
            list_2 = []
            list_3 = []

        list = list_0 + list_1 + list_2 + list_3

        """# perfect trajectory
        list_0 = [2]*18
        list_1 = [0]*12
        list_2 = [1]*(T)
        list = list_0 + list_1 + list_2"""

        for i in range(T):
            action = self.agent.act(obs)
            action = list[i]
            obs, reward, done, _ = self.env.step(action)
            sum_r = sum_r + reward

            self.agent.observe(obs, reward, done, False) #False is b.c. termination via time is handeled by environment
            df = pd.DataFrame([[self.agent.explorer.epsilon]])
            df.to_csv('process' + str(yaml_p['process_nr']).zfill(5) + '/log_agent.csv', mode='a', header=False, index=False)

            if render:
                self.env.render(mode=True)

            if done:
                break

        return sum_r
