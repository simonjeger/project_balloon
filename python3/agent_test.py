from analysis import plot_reward, plot_path, plot_2d_path, plot_3d_path, dist_hist, tuning, write_overview, clear
from build_environment import balloon3d
from build_agent import Agent

import numpy as np
import gym
import matplotlib.pyplot as plt
import torch
import time

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

from torch.utils.tensorboard import SummaryWriter

# always clear out previous tests
clear('test')

# initialize logger
writer = SummaryWriter(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_test')

env = balloon3d(0,0,'test',writer)
ag = Agent(0,0,'test',env,writer)
ag.load(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/')

with ag.agent.eval_mode():
    for i in range(yaml_p['num_epochs_test']):
        if yaml_p['environment'] != 'python3':

            # make sure the communication file is in the right form
            data = {
                'action': 0,
                'target': [-10,-10,-10],
                'c': ag.env.character.c,
                'ceiling': ag.env.character.ceiling,
                'delta_f_up': ag.env.character.delta_f_up,
                'delta_f_down': ag.env.character.delta_f_down,
                'mass_total': ag.env.character.mass_total
                }
            ag.env.character.send(data)

            # wait for human to give the go
            start = 'no'
            while start != 'yes':
                print('write "yes" if you want to start the episode')
                start = input()

        log = ag.run_epoch()
        print('epoch: ' + str(int(i)) + ' reward: ' + str(log))

time.sleep(100) #make sure the writing of tensorboard files is done
if yaml_p['overview']:
    write_overview()

plot_2d_path()
dist_hist()

if yaml_p['mode'] == 'tuning':
    tuning()
