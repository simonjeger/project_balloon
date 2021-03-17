import torch
import numpy as np
import copy
import torch.nn.functional as F
from collections import deque
import random
from log_utils import logger, mean_val
import os
import pandas as pd

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

class NN(torch.nn.Module):
    def __init__(self,in_dim,out_dim,n_hid):
        super(QNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_hid = n_hid

        self.fc1 = torch.nn.Linear(in_dim,n_hid,'relu')
        self.fc2 = torch.nn.Linear(n_hid,out_dim,'linear')

    def forward(self,x):
        x = F.relu(self.fc1(x))
        y = self.fc2(x)
        return y

class VAE:
    def __init__(self, wind):
        self.wind = wind
        acts = env.action_space
        self.model = QNet(obs.shape[0],acts.n,64)
        self.target_model = copy.deepcopy(self.model)
        self.rnd = RND(obs.shape[0],64,124)
        self.gamma = yaml_p['gamma']
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=yaml_p['lr']) #used to be 1e-3
        self.batch_size = 64
        self.buffer_size = yaml_p['buffer_size']
        self.step_counter = 0
        self.epsi_high = yaml_p['epsi_high']
        self.epsi_low = yaml_p['epsi_low']
        self.steps = 0
        self.count = 0
        self.decay = yaml_p['decay']
        self.eps = self.epsi_high
        self.update_target_step = yaml_p['update_target_step']
        self.log = logger()
        self.log.add_log('real_return')
        self.log.add_log('combined_return')
        self.log.add_log('avg_loss')

        self.replay_buffer = deque(maxlen=self.buffer_size)

        # initialize log file
        if os.path.isfile('process' + str(yaml_p['process_nr']).zfill(5) + '/log_agent.csv'):
            os.remove('process' + str(yaml_p['process_nr']).zfill(5) + '/log_agent.csv')

    def run_episode(self):
        obs = self.env.reset()
        sum_r = 0
        sum_tot_r = 0
        mean_loss = mean_val()

        t = 0
        while True:
            self.steps += 1
            self.eps = self.epsi_low + (self.epsi_high-self.epsi_low) * (np.exp(-1.0 * self.steps/self.decay))
            state = torch.Tensor(obs).unsqueeze(0)
            Q = self.model(state)
            num = np.random.rand()
            if (num < self.eps):
                action = torch.randint(0,Q.shape[1],(1,)).type(torch.LongTensor)
            else:
                action = torch.argmax(Q,dim=1)
            new_state, reward, done, info = self.env.step((action.item()))
            sum_r = sum_r + reward
            reward_i = self.rnd.get_reward(state).detach().clamp(-1.0,1.0).item()
            combined_reward = reward + reward_i
            sum_tot_r += combined_reward

            self.replay_buffer.append([obs,action,combined_reward,new_state,done])
            loss = self.update_model()
            mean_loss.append(loss)
            obs = new_state

            self.step_counter = self.step_counter + 1
            if (self.step_counter > self.update_target_step):
                self.target_model.load_state_dict(self.model.state_dict())
                self.step_counter = 0
                print('updated target model')

            df = pd.DataFrame([[self.eps]])
            df.to_csv('process' + str(yaml_p['process_nr']).zfill(5) + '/log_agent.csv', mode='a', header=False, index=False)

            #self.env.render(mode=True)

            if done:
                break

        self.log.add_item('real_return',sum_r)
        self.log.add_item('combined_return',sum_tot_r)
        self.log.add_item('avg_loss',mean_loss.get())

    def update_model(self):
        self.optimizer.zero_grad()
        num = len(self.replay_buffer)
        K = np.min([num,self.batch_size])
        samples = random.sample(self.replay_buffer, K)

        S0, A0, R1, S1, D1 = zip(*samples)
        S0 = torch.tensor( S0, dtype=torch.float)
        A0 = torch.tensor( A0, dtype=torch.long).view(K, -1)
        R1 = torch.tensor( R1, dtype=torch.float).view(K, -1)
        S1 = torch.tensor( S1, dtype=torch.float)
        D1 = torch.tensor( D1, dtype=torch.float)

        Ri = self.rnd.get_reward(S0)
        self.rnd.update(Ri)
        target_q = R1.squeeze() + self.gamma*self.target_model(S1).max(dim=1)[0].detach()*(1 - D1)
        policy_q = self.model(S0).gather(1,A0)
        L = F.smooth_l1_loss(policy_q.squeeze(),target_q.squeeze())
        L.backward()
        self.optimizer.step()
        return L.detach().item()

    def run_epoch(self):
        self.run_episode()
        return self.log

    def save_weights(self, path):
        torch.save(self.model.state_dict(), path + 'dqn_weights.h5f')
        torch.save(self.rnd.model.state_dict(), path + 'rnd_weights.h5f')

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path + 'dqn_weights.h5f'))
        self.target_model = copy.deepcopy(self.model)
        self.rnd.model.load_state_dict(torch.load(path + 'rnd_weights.h5f'))
