import torch
import numpy as np
from collections import deque
import os
import pandas as pd
import pfrl

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
        acts = env.action_space
        obs = env.observation_space
        self.qfunction = QFunction(obs.shape[0],acts.n)

        optimizer = torch.optim.Adam(self.qfunction.parameters(),lr=yaml_p['lr']) #used to be 1e-2
        gamma = yaml_p['gamma']
        replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=yaml_p['buffer_size'])

        epsi_high = yaml_p['epsi_high']
        epsi_low = yaml_p['epsi_low']
        decay = yaml_p['decay']
        explorer = pfrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=epsi_high, end_epsilon=epsi_low, decay_steps=decay, random_action_func=env.action_space.sample)

        phi = lambda x: x.astype(np.float32, copy=False)
        gpu = -1

        self.agent = pfrl.agents.DQN(
            self.qfunction,
            optimizer, #in my case ADAMS
            replay_buffer, #number of experiences I train my NN with
            gamma, #discount factor
            explorer, #how to choose next action
            replay_start_size=500, #number of experiences in replay buffer when training begins
            update_interval=1,
            target_update_interval=5, #taget network is copy of QFunction that is held constant to serve as a stable target for learning for fixed number of timesteps
            phi=phi,
            gpu=gpu, #actual GPU used for computation
        )

        # initialize log file
        if os.path.isfile('process' + str(yaml_p['process_nr']).zfill(5) + '/log_agent.csv'):
            os.remove('process' + str(yaml_p['process_nr']).zfill(5) + '/log_agent.csv')

    def run_epoch(self, render):
        obs = self.env.reset()
        sum_r = 0

        while True:
            action = self.agent.act(obs)
            new_state, reward, done, _ = self.env.step(action)
            sum_r = sum_r + reward

            self.agent.observe(new_state, reward, done, False)

            df = pd.DataFrame([[self.agent.explorer.epsilon]])
            df.to_csv('process' + str(yaml_p['process_nr']).zfill(5) + '/log_agent.csv', mode='a', header=False, index=False)

            if render:
                self.env.render(mode=True)

            if done:
                print('reward: ' + str(sum_r))
                break

    def save_weights(self, path):
        self.agent.save(path + 'weights_agent')

    def load_weights(self, path):
        self.agent.load(path + 'weights_agent')

    def visualize_q_map(self):
        res = 1

        Q_vis = np.zeros((self.env.size_x*res, self.env.size_z*res,4))
        for i in range(self.env.size_x*res):
            for j in range(self.env.size_z*res):
                position = np.array([i/10,j/10])
                obs = self.env.character_v(position)
                state = torch.Tensor(obs).unsqueeze(0)
                Q = self.qfunction.forward(state)

                Q_vis[i,j,0] = Q.q_values[0][0]
                Q_vis[i,j,1] = Q.q_values[0][1]
                Q_vis[i,j,2] = Q.q_values[0][2]
                Q_vis[i,j,3] = Q.greedy_actions

        return Q_vis
