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
    def __init__(self, train_or_test, env, writer=None):
        self.train_or_test = train_or_test
        self.env = env
        self.stash = [0]*yaml_p['phase']
        self.writer = writer

        acts = env.action_space
        obs = env.observation_space

        if yaml_p['continuous']:
            obs_size = obs.low.size
            action_size = acts.low.size

            def squashed_diagonal_gaussian_head(x):
                assert x.shape[-1] == action_size * 2
                mean, log_scale = torch.chunk(x, 2, dim=1)
                log_scale = torch.clamp(log_scale, -20.0, 2.0)
                var = torch.exp(log_scale * 2)
                base_distribution = torch.distributions.Independent(
                    torch.distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
                )
                # cache_size=1 is required for numerical stability
                return torch.distributions.transformed_distribution.TransformedDistribution(
                    base_distribution, [torch.distributions.transforms.TanhTransform(cache_size=1)]
                )

            policy = torch.nn.Sequential(
                torch.nn.Linear(obs_size, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, action_size * 2),
                pfrl.nn.lmbda.Lambda(squashed_diagonal_gaussian_head),
            )

            torch.nn.init.xavier_uniform_(policy[0].weight)
            torch.nn.init.xavier_uniform_(policy[2].weight)
            torch.nn.init.xavier_uniform_(policy[4].weight)
            policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

            def make_q_func_with_optimizer():
                q_func = torch.nn.Sequential(
                    pfrl.nn.ConcatObsAndAction(),
                    torch.nn.Linear(obs_size + action_size, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 1),
                )
                torch.nn.init.xavier_uniform_(q_func[1].weight)
                torch.nn.init.xavier_uniform_(q_func[3].weight)
                torch.nn.init.xavier_uniform_(q_func[5].weight)
                q_func_optimizer = torch.optim.Adam(q_func.parameters(), lr=3e-4)
                return q_func, q_func_optimizer

            q_func1, q_func1_optimizer = make_q_func_with_optimizer()
            q_func2, q_func2_optimizer = make_q_func_with_optimizer()

            def burnin_action_func():
                return np.random.uniform(acts.low, acts.high).astype(np.float32) #select random actions until model is updated one or more times

            if yaml_p['agent_type'] == 'SoftActorCritic':
                self.agent = pfrl.agents.SoftActorCritic(
                    policy,
                    q_func1,
                    q_func2,
                    policy_optimizer,
                    q_func1_optimizer,
                    q_func2_optimizer,
                    pfrl.replay_buffers.ReplayBuffer(capacity=yaml_p['buffer_size']),
                    gamma=0.95,
                    replay_start_size=yaml_p['replay_start_size'],
                    gpu=-1,
                    minibatch_size=yaml_p['minibatch_size'],
                    burnin_action_func=burnin_action_func,
                    entropy_target=-action_size,
                    temperature_optimizer_lr=3e-4,
                )

            else:
                print('please choose one of the implemented agents')

        else:
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

            if yaml_p['agent_type'] == 'DoubleDQN':
                self.qfunction = QFunction(obs.shape[0],acts.n)

                self.agent = pfrl.agents.DoubleDQN(
                    self.qfunction,
                    torch.optim.Adam(self.qfunction.parameters(),lr=yaml_p['lr']), #in my case ADAMS
                    #torch.optim.Adadelta(self.qfunction.parameters()),
                    pfrl.replay_buffers.ReplayBuffer(capacity=yaml_p['buffer_size']), #number of experiences I train my NN with
                    yaml_p['gamma'], #discount factor
                    explorer, #how to choose next action
                    clip_delta=True,
                    max_grad_norm=yaml_p['max_grad_norm'],
                    replay_start_size=yaml_p['replay_start_size'], #number of experiences in replay buffer when training begins
                    update_interval=yaml_p['update_interval'], #in later parts of the code I set the timer to this, so it updates every episode
                    target_update_interval=yaml_p['target_update_interval'],
                    minibatch_size=yaml_p['minibatch_size'], #minibatch_size used for training the q-function network
                    n_times_update=yaml_p['n_times_update'], #how many times we update the NN with a new batch per update step
                    phi=lambda x: x.astype(np.float32, copy=False), #feature extractor applied to observations
                    gpu=-1, #actual GPU used for computation
                )

            else:
                print('please choose one of the implemented agents')

        self.epi_n = 0
        self.step_n = 0

    def set_reachable_target(self):
        curr_start = 0.3
        curr_window = 0.4
        curr_end = 1 - curr_window
        if self.train_or_test == 'train':
            curr = curr_start + (curr_end - curr_start)*min(self.step_n/yaml_p['curriculum'],1)
        else:
            curr = curr_end

        round = 0
        while True:
            self.env.character.target = [-10,-10] #set target outside map

            if self.env.character.position[1] < self.env.size_z*0.2:
                action = np.random.normal(1.8,0.3)
            elif self.env.character.position[1] > self.env.size_z*0.8:
                action = np.random.normal(0.3,0.3)
            else:
                action = np.random.normal(1,0.3)
            action = np.clip(action,0,2)

            # actions are not in the same range in discrete / continuous cases
            if yaml_p['continuous']:
                action = action
            else:
                action = np.round(action,0)

            _, _, done, _ = self.env.step(action, blind=True)

            dist_to_start = np.sqrt((self.env.character.position[0] - self.env.character.start[0])**2)
            if done & (dist_to_start > yaml_p['radius']):
                break
            elif done:
                if round >= 3:
                    break
                else:
                    self.env.reset(blind=True)
                    round += 1

        idx = np.random.randint(int(curr*len(self.env.character.path)), max(int((curr+curr_window-0.1)*len(self.env.character.path)),int(curr*len(self.env.character.path)) + 1))
        target = self.env.character.path[idx]
        self.env.reset(blind=True)
        self.env.character.target = target

    def run_epoch(self, render):
        obs = self.env.reset()
        sum_r = 0

        if yaml_p['curriculum'] > 0: #reset target to something reachable if that flag is set
            self.set_reachable_target()

        while True:
            action = self.agent.act(obs) #uses self.agent.model to decide next step

            # actions are not in the same range in discrete / continuous cases
            if yaml_p['continuous']:
                action = action[0]+1
            else:
                action = action

            obs, reward, done, _ = self.env.step(action)

            sum_r = sum_r + reward
            self.agent.observe(obs, reward, done, False) #False is b.c. termination via time is handeled by environment

            self.step_n += 1

            if render:
                self.env.render(mode=True)

            if done:
                # logger
                if self.writer is not None:
                    if yaml_p['continuous']:
                        self.writer.add_scalar('epsilon', 0 , self.step_n-1) # because we do above self.step_n += 1
                        self.writer.add_scalar('loss_qfunction', 0, self.step_n-1)
                    else:
                        if yaml_p['explorer_type'] == 'LinearDecayEpsilonGreedy':
                            self.writer.add_scalar('epsilon', self.agent.explorer.epsilon , self.step_n-1) # because we do above self.step_n += 1
                        if len(self.agent.loss_record) != 0:
                            self.writer.add_scalar('loss_qfunction', np.mean(self.agent.loss_record), self.step_n-1)
                        """
                        for i in range(len(self.agent.loss_record)):
                            self.writer.add_scalar('local_loss_qfunction', self.agent.loss_record[i], self.step_n-1 + i/len(self.agent.loss_record))
                        """
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

        if self.writer is not None:
            self.writer.add_scalar('weights_saved', self.epi_n , self.step_n-1) # because we do above self.step_n += 1

    def load_weights(self, path):
        self.agent.load(path + 'weights_agent')
        print('weights loaded')

    def visualize_q_map(self):
        res_x = max(10/self.env.size_x,0.1)
        res_z = max(10/self.env.size_z,0.1)

        Q_vis = np.zeros((int(self.env.size_x*res_x), int(self.env.size_z*res_z),8))
        for i in range(int(self.env.size_x*res_x)):
            for j in range(int(self.env.size_z*res_z)):
                position = np.array([(i+0.5)/res_x,(j+0.5)/res_z])
                obs = self.env.character_v(position)
                state = torch.Tensor(obs).unsqueeze(0)
                Q = self.agent.model.forward(state)
                Q_tar = self.agent.target_model.forward(state)

                # model
                Q_vis[i,j,0] = Q.q_values[0][0]
                Q_vis[i,j,1] = Q.q_values[0][1]
                Q_vis[i,j,2] = Q.q_values[0][2]
                Q_vis[i,j,3] = Q.greedy_actions

                """
                # target model
                Q_vis[i,j,4] = Q_tar.q_values[0][0]
                Q_vis[i,j,5] = Q_tar.q_values[0][1]
                Q_vis[i,j,6] = Q_tar.q_values[0][2]
                Q_vis[i,j,7] = Q_tar.greedy_actions
                """

        return Q_vis
