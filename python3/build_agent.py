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
from scipy.ndimage import gaussian_filter
import alphashape
from shapely.geometry import Point
import matplotlib.pyplot as plt
from descartes import PolygonPatch
import pickle
import logging
import pygame
from sys import exit
import time

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
    def __init__(self, epi_n, step_n, train_or_test, env, writer=None):
        self.train_or_test = train_or_test
        self.env = env
        self.writer = writer
        self.seed = 0

        self.clip = 0.01

        acts = env.action_space
        obs = env.observation_space

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

        width = yaml_p['width']
        depth = yaml_p['depth']

        modules = []
        modules.append(torch.nn.Linear(obs_size, width))
        modules.append(torch.nn.ReLU())
        for _ in range(depth):
            modules.append(torch.nn.Linear(width, width))
            modules.append(torch.nn.ReLU())
        modules.append(torch.nn.Linear(width, action_size*2))
        modules.append(pfrl.nn.lmbda.Lambda(squashed_diagonal_gaussian_head))
        policy = torch.nn.Sequential(*modules)

        for i in range(depth+2):
            torch.nn.init.xavier_uniform_(policy[i*2].weight)

        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=yaml_p['lr'])
        self.scheduler_policy = torch.optim.lr_scheduler.StepLR(policy_optimizer, step_size=yaml_p['lr_scheduler'], gamma=0.1, verbose=False)
        self.scheduler_policy._step_count = step_n

        def make_q_func_with_optimizer():
            width = yaml_p['width']
            depth = yaml_p['depth']

            modules = []
            modules.append(pfrl.nn.ConcatObsAndAction())
            modules.append(torch.nn.Linear(obs_size + action_size, width))
            modules.append(torch.nn.ReLU())
            for _ in range(depth):
                modules.append(torch.nn.Linear(width, width))
                modules.append(torch.nn.ReLU())
            modules.append(torch.nn.Linear(width, 1))
            q_func = torch.nn.Sequential(*modules)

            for i in range(depth+2):
                torch.nn.init.xavier_uniform_(q_func[i*2+1].weight)

            q_func_optimizer = torch.optim.Adam(q_func.parameters(), lr=yaml_p['lr'])
            self.scheduler_qfunc = torch.optim.lr_scheduler.StepLR(q_func_optimizer, step_size=yaml_p['lr_scheduler'], gamma=0.1, verbose=False)
            self.scheduler_qfunc._step_count = step_n
            return q_func, q_func_optimizer

        q_func1, q_func1_optimizer = make_q_func_with_optimizer()
        q_func2, q_func2_optimizer = make_q_func_with_optimizer()

        self.action_burnin = None

        def burnin_action_func(type=yaml_p['burnin']):
            if type == 'advanced':
                if self.action_burnin is None:
                    self.action_burnin = np.random.uniform(self.clip,1-self.clip)
                elif abs(self.env.character.velocity[2]*yaml_p['unit_z']) < 0.1: #x m/s, basically: did I reach the set altitude?
                    if np.random.uniform(0,yaml_p['alt_resample']) < yaml_p['delta_t']: # if yes, chances are N/delta_t that I choose a new altitude
                        self.action_burnin = np.random.uniform(self.clip,1-self.clip)
            elif type == 'basic':
                self.action_burnin = np.random.uniform(self.clip,1-self.clip)
            return [self.action_burnin]

        if torch.cuda.is_available():
            device = 0
        else:
            device = -1

        #initialize soft actor critic agent
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
            gpu=device,
            minibatch_size=yaml_p['minibatch_size'],
            update_interval=yaml_p['update_interval'],
            burnin_action_func=burnin_action_func,
            entropy_target=-action_size,
            #temperature_optimizer_lr=yaml_p['temperature_optimizer_lr']
        )

        self.global_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=yaml_p['buffer_size'])
        self.old_buffer_size = 0

        self.epi_n = epi_n
        self.step_n = step_n
        self.render_ratio = yaml_p['unit_xy'] / yaml_p['unit_z']

    def run_epoch(self,importance=None):
        obs = self.env.reset()
        sum_r = 0

        if (yaml_p['reachability_study'] > 0):
            obs = self.reachability_study()
        elif yaml_p['set_reachable_target']:
            obs = self.set_reachable_target()

        if importance is not None:
            self.env.character.importance = importance

        if yaml_p['mode'] == 'game':
            # If it's the beginning of a new round
            action = self.env.render_machine.load_screen()

        self.HER_obs = [obs]
        self.HER_pos = [self.env.character.start]
        self.HER_action = []
        self.HER_proj_action = []
        self.HER_U = []
        self.HER_reward = []
        self.HER_done = []
        self.HER_target = []
        self.HER_residual = []

        while True:
            if yaml_p['render']:
                self.env.render(mode=True)

            if yaml_p['mode'] == 'reinforcement_learning':
                action_RL = self.agent.act(obs) #uses self.agent.model to decide next step
                action = np.clip(action_RL[0],self.clip,1-self.clip) #gym sometimes violates the conditions set in the environment

            elif yaml_p['mode'] == 'game':
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == 60:
                            action = 0
                        elif event.key == pygame.K_1:
                            action = 0.1
                        elif event.key == pygame.K_2:
                            action = 0.2
                        elif event.key == pygame.K_3:
                            action = 0.3
                        elif event.key == pygame.K_4:
                            action = 0.4
                        elif event.key == pygame.K_5:
                            action = 0.5
                        elif event.key == pygame.K_6:
                            action = 0.6
                        elif event.key == pygame.K_7:
                            action = 0.7
                        elif event.key == pygame.K_8:
                            action = 0.8
                        elif event.key == pygame.K_9:
                            action = 0.9
                        elif event.key == pygame.K_0:
                            action = 1

                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()

                _ = self.agent.act(obs) #this is only so it works in training mode
                action = np.clip(action,self.clip,1-self.clip)
                action_RL = action #this is only so it works with HER

            elif yaml_p['mode'] == 'simple':
                _ = self.agent.act(obs) #this is only so it works in training mode
                action = self.act_simple(self.env.character)
                action_RL = action #this is only so it works with HER

            elif yaml_p['mode'] == 'hybrid':
                action_RL = self.agent.act(obs) #this is only so it works in training mode
                action = self.act_simple(self.env.character, action_RL[0])

            else:
                print('ERROR: Please choose one of the available modes.')

            obs, reward, done, _ = self.env.step(action) #I just need to pass a target that is not None for the logger to kick in
            sum_r = sum_r + reward
            self.agent.observe(obs, reward, done, False) #False is b.c. termination via time is handeled by environment
            self.step_n += 1
            self.scheduler_policy.step()
            self.scheduler_qfunc.step()

            # for hindsight experience replay
            self.HER_obs.append(obs)
            self.HER_pos.append(copy.copy(self.env.character.position))
            self.HER_action.append(action_RL)
            self.HER_proj_action.append(self.env.character.proj_action(self.env.character.position, self.env.character.target))
            self.HER_U.append(copy.copy(self.env.character.U))
            self.HER_target.append(copy.copy(self.env.character.target))
            self.HER_residual.append(copy.copy(self.env.character.residual))

            if done:
                if yaml_p['render']:
                    self.env.render(mode=True)

                # logger
                if self.writer is not None:
                    if (len(self.agent.q_func1_loss_record) > 0):
                        self.writer.add_scalar('q_func1_loss', self.agent.q_func1_loss_record[-1], self.step_n-1)
                        self.writer.add_scalar('q_func2_loss', self.agent.q_func2_loss_record[-1], self.step_n-1)
                    self.writer.add_scalar('buffer_len', len(self.agent.replay_buffer.memory), self.step_n-1)
                    self.writer.add_scalar('scheduler_policy', self.scheduler_policy.get_last_lr()[0], self.step_n-1)
                    self.writer.add_scalar('scheduler_qfunc', self.scheduler_qfunc.get_last_lr()[0], self.step_n-1)

                self.epi_n += 1
                break

        if yaml_p['HER'] & (self.train_or_test == 'train'):
            idx = self.idx_on_path(self.env.character.path, self.env.character.start)
            target = self.env.character.path[idx] #set target at last position that was reached, but still within bounds
            self.HER(target)

        # mark in map_test if this was a success or not
        if (yaml_p['reachability_study'] > 0) & (self.train_or_test == 'test'):
            self.map_test()

        return sum_r

    def reachability_study(self):
        self.path_mt_pkl = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/map_test/' + self.env.world_name + '.pkl'
        self.path_mt_png = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/map_test/' + self.env.world_name + '.png'
        self.path_rs_pkl = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/reachability_study/' + self.env.world_name + '.pkl'
        self.path_rs_as = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/reachability_study/' + self.env.world_name + '_as' +'.pkl'
        self.path_rs_png = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/reachability_study/' + self.env.world_name + '.png'
        self.path_rs_csv = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/reachability_study/percentage' + '.csv'

        if os.path.isfile(self.path_rs_as):
            with open(self.path_rs_as,'rb') as fid:
                alpha_shape = pickle.load(fid)
        else:
            fig, ax = plt.subplots()
            ax.scatter(0,0,color='white')
            ax.scatter(self.env.size_x,self.env.size_y,color='white')

            x_global = []
            y_global = []

            res = 10
            for i in range(yaml_p['reachability_study']):
                self.random_roll_out()
                self.env.reset(target=[-10,-10,-10])

                x = []
                y = []

                for j in range(res+1):
                    if j <= res:
                        k = int(j*len(self.env.path_reachability[-1][-1])/res - 1)
                    else:
                        k = len(self.env.path_reachability[-1][-1]) - 1 #this is to make sure the last point is considered
                    x_j = self.env.path_reachability[-1][-1][k][0]
                    y_j = self.env.path_reachability[-1][-1][k][1]

                    x.append(x_j)
                    y.append(y_j)
                    x_global.append(x_j)
                    y_global.append(y_j)
                ax.plot(x,y,color='grey')
                print('reachability_study: ' + str(np.round(i/yaml_p['reachability_study']*100,0)) + ' %')

            points = list(zip(x_global,y_global))

            # Generate the alpha shape
            alpha = 0.2

            logging.disable() #to suppress: WARNING:root:Singular matrix. Likely caused by all points lying in an N-1 space.
            alpha_shape = alphashape.alphashape(points, alpha)
            print('alpha_shape calculated')

            # save alpha_shape for next time
            with open(self.path_rs_as,'wb') as fid:
                pickle.dump(alpha_shape, fid)

            # Plot alpha shape
            ax.add_patch(PolygonPatch(alpha_shape, fc='blue', ec='blue'))
            ax.set_title(str(np.round(alpha_shape.area/(self.env.size_x*self.env.size_y)*100,2)) + '% reachable')
            ax.set_aspect('equal')
            plt.savefig(self.path_rs_png)

            # Save pickled version for later edits
            with open(self.path_rs_pkl,'wb') as fid:
                pickle.dump(ax, fid)

            plt.close()

            df = pd.DataFrame([alpha_shape.area/(self.env.size_x*self.env.size_y)],[self.env.world_name])
            df.to_csv(self.path_rs_csv, mode='a', header=False)

        # Place target within shape
        target = [-10,-10,-10]
        while alpha_shape.contains(Point(target[0],target[1])) == False:
            if self.train_or_test == 'test':
                np.random.seed(self.seed)
                self.seed += 1
            target = [np.random.uniform(0,self.env.size_x), np.random.uniform(0,self.env.size_z), np.random.uniform(0,self.env.character.ceiling)]
            obs = self.env.reset(target=target)
            return obs

    def set_reachable_target(self):
        self.random_roll_out()
        idx = self.idx_on_path(self.env.character.path, self.env.character.start)

        self.env.path_roll_out = self.env.character.path[0:idx]
        target = self.env.character.path[idx]

        self.env.reward_roll_out = sum(self.env.reward_list[0:int(idx/self.env.character.n)]) + 1 #because the physics simmulation takes n timesteps)
        obs = self.env.reset(target=target)
        return obs

    def random_roll_out(self):
        round = 0

        if self.train_or_test == 'test':
            np.random.seed(self.seed)
            self.seed += 1
        action = np.random.uniform(0.1,0.9)

        while True:
            self.env.character.target = [-10,-10,-10] #set target outside map

            if self.train_or_test == 'test':
                np.random.seed(self.seed)
                self.seed += 1
            action = self.agent.burnin_action_func(type='advanced')[0]
            _, _, done, _ = self.env.step(action,skip=True)
            #self.env.render(mode=True)
            sucess = False

            if done:
                break

        # write down path for reachability study
        self.env.path_reachability.append([self.env.character.path])

    def idx_on_path(self, path, start):
        # write down path and set target
        coord_x = []
        coord_y = []
        for i in range(len(path)):
            coord_x.append(path[i][0])
            coord_y.append(path[i][1])

        for _ in range(100): # if I can't find anything that's far enough from the start after n tries, just take the last one
            if self.train_or_test == 'test':
                np.random.seed(self.seed)
                self.seed += 1
            idx_x = np.random.uniform(min(coord_x),max(coord_x))
            idx_y = np.random.uniform(min(coord_y),max(coord_y))
            idx = np.argmin(np.sqrt(np.subtract(coord_x,idx_x)**2 + np.subtract(coord_y,idx_y)**2))

            target = path[idx]

            not_to_close = np.sqrt((target[0] - start[0])**2 + (target[1] - start[1])**2) > yaml_p['radius_xy']*yaml_p['unit_z']/yaml_p['unit_xy']*4 #I want to be at least 4 radius away from the start
            not_out_of_bounds_x = (0 < target[0]) & (target[0] < self.env.size_x - yaml_p['radius_xy']/yaml_p['unit_xy']*yaml_p['unit_z'])
            not_out_of_bounds_y = (0 < target[1]) & (target[1] < self.env.size_y - yaml_p['radius_xy']/yaml_p['unit_xy']*yaml_p['unit_z'])

            if not_to_close & not_out_of_bounds_x & not_out_of_bounds_y:
                break

        return idx

    def map_test(self):
        if os.path.isfile(self.path_mt_pkl):
            with open(self.path_mt_pkl,'rb') as fid:
                ax = pickle.load(fid)
        else:
            with open(self.path_rs_pkl,'rb') as fid:
                ax = pickle.load(fid)

        if self.env.success:
            color = 'green'
        else:
            color = 'red'

        circle = plt.Circle((self.env.character.target[0], self.env.character.target[1]),yaml_p['radius_xy']*yaml_p['unit_z']/yaml_p['unit_xy'], color=color, fill=False, zorder=np.inf) #always plot on top
        ax.add_patch(circle)
        plt.savefig(self.path_mt_png)

        # Save pickled version for later edits
        with open(self.path_mt_pkl,'wb') as fid:
            pickle.dump(ax, fid)
        plt.close()

    def HER(self, target):
        self.env.character.t = 1 #it can't be zero, otherwise the cost function thinks we are out of time

        tar_x = int(np.clip(target[0],0,self.env.size_x - 1))
        tar_y = int(np.clip(target[1],0,self.env.size_y - 1))
        tar_z_squished = (target[2]-self.env.character.world[0,tar_x,tar_y,0])/(self.env.character.ceiling - self.env.character.world[0,tar_x,tar_y,0])

        # fix all the state spaces
        min_proj_dist = np.inf
        for i in range(len(self.HER_obs)):
            position = self.HER_pos[i]

            pos_z_squished = self.HER_obs[i][8]
            residual = target - position

            #this is only an approximation because I don't look at all the points
            min_proj_dist_prop = np.sqrt((residual[0]*self.render_ratio/self.env.character.radius_xy)**2 + (residual[1]*self.render_ratio/self.env.character.radius_xy)**2 + (residual[2]/self.env.character.radius_z)**2)
            if min_proj_dist > min_proj_dist_prop:
                min_proj_dist = min_proj_dist_prop

            self.HER_obs[i][0:3] = np.append(self.env.character.normalize_map(residual[0:2]), [tar_z_squished - pos_z_squished])

            if i > 0:
                in_bounds = True
                if (self.HER_pos[i][0] < 0) | (self.HER_pos[i][0] > self.env.size_x - 1):
                    in_bounds = False
                if (self.HER_pos[i][1] < 0) | (self.HER_pos[i][1] > self.env.size_y - 1):
                    in_bounds = False
                if (self.HER_pos[i][2] < 0) | (self.HER_pos[i][2] > self.env.size_z - 1): #not totally complete, because terrain and ceiling
                    in_bounds = False
                reward, done, success = self.env.cost(self.env.character.start, position, target, self.HER_action[i-1][0], self.HER_U[i-1], min_proj_dist, in_bounds) #is slightly off on the part with the proj_action because the world is time depentent
                self.HER_reward.append(reward)
                self.HER_done.append(done)

        if not self.HER_done[-1]:
            self.HER_done[-1] = True

        # put them into the buffer
        i = 0
        while len(self.HER_done) > 1:
            self.agent.t += 1
            self.agent.replay_buffer.append(state=self.HER_obs[i],
                    action=self.HER_action[i],
                    reward=self.HER_reward[i],
                    next_state=self.HER_obs[i+1],
                    next_action=None,
                    is_state_terminal=self.HER_done[i],
                    env_id=0) #I don't fully understand what this does, but for the normal buffer it is = 0 as well

            if self.HER_done[i]:
                self.agent.replay_buffer.stop_current_episode(env_id=0)
                self.agent.replay_updater.update_if_necessary(self.agent.t)
                break
            i += 1

    def weights_init(self, m): #currently not in use
        if isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, 0.0001)

    def save_weights(self, path):
        self.agent.save(path + 'weights_agent')

        self.save_buffer(path)
        self.load_buffer()

        print('weights and buffer saved at ' + time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime()))

    def load_weights(self, path):
        self.agent.load(path + 'weights_agent')

        if self.train_or_test == 'train':
            self.save_buffer(path)
            self.load_buffer()

        print('weights and buffer loaded at ' + time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime()))

    def save_buffer(self, path):
        if len(self.agent.replay_buffer.memory) > 0:
            self.wait('save')

            local_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=int(yaml_p['buffer_size']))
            if os.path.isfile(path + 'buffer'):
                local_buffer.load(path + 'buffer')
            for i in range(self.old_buffer_size,len(self.agent.replay_buffer.memory)):
                local_buffer.memory.append(self.agent.replay_buffer.memory[i])
            local_buffer.save(path + 'buffer')

    def load_buffer(self):
        self.wait('load')
        start = yaml_p['global_buffer_nr']
        end = start + yaml_p['global_buffer_N']

        i_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=int(yaml_p['buffer_size']/yaml_p['global_buffer_N']))
        local_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=int(yaml_p['buffer_size']))
        for i in range(start,end):
            path = str(i).zfill(5) + '/'
            if os.path.isfile(path + 'buffer'):
                i_buffer.load(path + 'buffer')
                local_buffer.memory.extend(i_buffer.memory)

        self.agent.replay_buffer.memory = copy.copy(local_buffer.memory)
        self.old_buffer_size = len(self.agent.replay_buffer.memory)

    def wait(self, save_or_load):
        save = 2 #s
        pause = 60 #s
        load = 2 #s
        cycle = save + pause + load
        if save_or_load == 'save':
            while int(time.time()%cycle) >= save:
                time.sleep(1)
        elif save_or_load == 'load':
            while int(time.time()%cycle) <= save + pause:
                time.sleep(1)

    def act_simple(self, character, p=None):
        tar_x = int(np.clip(character.target[0],0,self.env.size_x-1))
        tar_y = int(np.clip(character.target[1],0,self.env.size_y-1))
        tar_z = int(np.clip(character.target[2],0,self.env.size_z-1))
        residual_x = character.residual[0]
        residual_y = character.residual[1]
        residual_z = character.residual[2]
        tar_z_squished = (character.target[2]-character.world[0,tar_x,tar_y,0])/(character.ceiling - character.world[0,tar_x,tar_y,0])
        vel_x = character.velocity[0]
        vel_y = character.velocity[1]

        if p is None:
            k_1 = 5 #5
            k_2 = 100 #100

            p_1 = 1/abs(residual_x)*k_1
            p_2 = 1/abs(residual_y)*k_1
            p_3 = np.clip(vel_x*residual_x*k_2,0.1,np.inf)
            p_4 = np.clip(vel_y*residual_y*k_2,0.1,np.inf)
            p = np.clip(p_1*p_2*p_3*p_4,0,1)

            p = np.round(p,0) #bang bang makes most sense here
            #p = 1 #switch to heuristic_without_wind
        else:
            p = p

        projections = self.env.character.proj_action(character.position, character.target)
        projections = gaussian_filter(projections,sigma=1)

        action = np.argmax(projections)/len(projections)*(1-p) + tar_z_squished*p
        action = np.clip(action,self.clip,1-self.clip) #avoid crashing into terrain

        return action
