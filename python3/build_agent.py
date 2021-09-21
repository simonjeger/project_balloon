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

        def burnin_action_func():
            return np.random.uniform(acts.low, acts.high).astype(np.float32) #select random actions until model is updated one or more times

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
            burnin_action_func=burnin_action_func,
            entropy_target=-action_size,
            temperature_optimizer_lr=yaml_p['temperature_optimizer_lr'],
        )

        self.epi_n = epi_n
        self.step_n = step_n
        self.render_ratio = yaml_p['unit_xy'] / yaml_p['unit_z']

    def run_epoch(self,importance=None):
        obs = self.env.reset()
        sum_r = 0

        if (yaml_p['reachability_study'] > 0):
            self.reachability_study()
        elif yaml_p['set_reachable_target']:
            self.set_reachable_target()

        if importance is not None:
            self.env.character.importance = importance

        if yaml_p['mode'] == 'game':
            # If it's the beginning of a new round
            action = self.env.render_machine.load_screen()

        while True:
            if yaml_p['render']:
                self.env.render(mode=True)

            if yaml_p['mode'] == 'reinforcement_learning':
                action = self.agent.act(obs) #uses self.agent.model to decide next step
                action = (action[0]+1)/2

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
                action = np.clip(action,0,1)

            elif yaml_p['mode'] == 'simple':
                _ = self.agent.act(obs) #this is only so it works in training mode
                action = self.act_simple(self.env.character)

            else:
                print('ERROR: Please choose one of the available modes.')

            obs, reward, done, _ = self.env.step(action)
            sum_r = sum_r + reward
            self.agent.observe(obs, reward, done, False) #False is b.c. termination via time is handeled by environment

            self.step_n += 1
            self.scheduler_policy.step()
            self.scheduler_qfunc.step()

            if done:
                if yaml_p['render']:
                    self.env.render(mode=True)

                # logger
                if self.writer is not None:
                    self.writer.add_scalar('epsilon', 0 , self.step_n-1) # because we do above self.step_n += 1
                    self.writer.add_scalar('loss_qfunction', 0, self.step_n-1)
                    self.writer.add_scalar('scheduler_policy', self.scheduler_policy.get_last_lr()[0], self.step_n-1)
                    self.writer.add_scalar('scheduler_qfunc', self.scheduler_qfunc.get_last_lr()[0], self.step_n-1)

                self.epi_n += 1
                break

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
                self.env.reset(keep_world=True)

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
            self.env.character.target = target

    def set_reachable_target(self):
        self.random_roll_out()

        # write down path and set target
        coord_x = []
        coord_y = []
        for i in range(len(self.env.character.path)):
            coord_x.append(self.env.character.path[i][0])
            coord_y.append(self.env.character.path[i][1])

        for _ in range(100): # if I can't find anything that's far enough from the start after n tries, just take the last one
            if self.train_or_test == 'test':
                np.random.seed(self.seed)
                self.seed += 1
            idx_x = np.random.uniform(min(coord_x),max(coord_x))
            idx_y = np.random.uniform(min(coord_y),max(coord_y))
            idx = np.argmin(np.sqrt(np.subtract(coord_x,idx_x)**2 + np.subtract(coord_y,idx_y)**2))

            target = self.env.character.path[idx]

            not_to_close = np.sqrt((target[0] - self.env.character.start[0])**2 + (target[0] - self.env.character.start[1])**2) > yaml_p['radius_xy']*yaml_p['unit_z']/yaml_p['unit_xy']
            not_out_of_bounds_x = (0 < target[0]) & (target[0] < self.env.size_x - yaml_p['radius_xy']/yaml_p['unit_xy']*yaml_p['unit_z'])
            not_out_of_bounds_y = (0 < target[1]) & (target[1] < self.env.size_y - yaml_p['radius_xy']/yaml_p['unit_xy']*yaml_p['unit_z'])

            if not_to_close & not_out_of_bounds_x & not_out_of_bounds_y:
                break

        self.env.path_roll_out = self.env.character.path[0:idx]
        target = self.env.character.path[idx]

        self.env.reward_roll_out = sum(self.env.reward_list[0:int(idx/self.env.character.n)]) + 1 #because the physics simmulation takes n timesteps)
        self.env.reset(keep_world=True)
        self.env.character.target = target

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

            if abs(self.env.character.velocity[2]*yaml_p['unit_z']) < 0.5: #x m/s, basically: did I reach the set altitude?
                if np.random.uniform() < 0.25: # if yes, set a new one with a certain probability
                    action = np.random.uniform(0.1,0.9)

            _, _, done, _ = self.env.step(action, keep_world=True)
            sucess = False

            if done:
                break

        # write down path for reachability study
        self.env.path_reachability.append([self.env.character.path])

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

    def weights_init(self, m):
        if isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, 0.0001)

    def save_weights(self, path):
        self.agent.save(path + 'weights_agent')

    def load_weights(self, path):
        self.agent.load(path + 'weights_agent')

    def act_simple(self, character):
        pos_x = int(np.clip(character.position[0],0,self.env.size_x-1))
        pos_y = int(np.clip(character.position[1],0,self.env.size_y-1))
        pos_z = int(np.clip(character.position[2],0,self.env.size_z-1))
        tar_x = int(np.clip(character.target[0],0,self.env.size_x-1))
        tar_y = int(np.clip(character.target[1],0,self.env.size_y-1))
        tar_z = int(np.clip(character.target[2],0,self.env.size_z-1))
        residual_x = character.residual[0]
        residual_y = character.residual[1]
        residual_z = character.residual[2]
        tar_z_squished = (character.target[2]-character.world[0,tar_x,tar_y,0])/(character.ceiling - character.world[0,tar_x,tar_y,0])
        vel_x = character.velocity[0]
        vel_y = character.velocity[1]

        # window_squished
        data = character.world
        res = self.env.size_z
        data_squished = np.zeros((len(data),self.env.size_x,self.env.size_y,res))
        for i in range(self.env.size_x):
            for j in range(self.env.size_y):
                bottom = data[0,i,j,0]
                top = character.ceiling

                x_old = np.arange(0,self.env.size_z,1)
                x_new = np.linspace(bottom,top,res)
                data_squished[0,:,:,:] = data[0,:,:,:] #terrain stays the same

                for k in range(1,len(data)):
                    data_squished[k,i,j,:] = np.interp(x_new,x_old,data[k,i,j,:])

        wind_x = data_squished[-4,pos_x,pos_y,:]
        wind_x = gaussian_filter(wind_x,sigma=1)

        wind_y = data_squished[-3,pos_x,pos_y,:]
        wind_y = gaussian_filter(wind_y,sigma=1)

        k_1 = 5 #5
        k_2 = 100 #100

        p_1 = 1/abs(residual_x)*k_1
        p_2 = 1/abs(residual_y)*k_1
        p_3 = np.clip(vel_x*residual_x*k_2,0.1,np.inf)
        p_4 = np.clip(vel_y*residual_y*k_2,0.1,np.inf)
        p = np.clip(p_1*p_2*p_3*p_4,0,1)

        p = np.round(p,0) #bang bang makes most sense here
        #p = 1 #switch to heuristic_without_wind

        norm_wind = np.sqrt(wind_x**2 + wind_y**2)
        projections = (residual_x*wind_x + residual_y*wind_y)/norm_wind
        action = np.argmax(projections)/len(projections)*(1-p) + tar_z_squished*p
        action = np.clip(action,0.05,1) #avoid crashing into terrain

        return action
