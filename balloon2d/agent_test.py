from analysis import plot_reward, plot_path, plot_qmap, write_overview, clear
from build_environment import balloon2d
from build_agent import Agent

import numpy as np
import gym
import matplotlib.pyplot as plt
import torch

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

env = balloon2d(0,0,'test',writer)
ag = Agent(0,0,'test',env,writer)
ag.load_weights(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/')

with ag.agent.eval_mode():
    for i in range(yaml_p['num_epochs_test']):
        log = ag.run_epoch(False)
        print('epoch: ' + str(i) + ' reward: ' + str(log))

# analyse
if yaml_p['overview']:
    write_overview()

# Delete log files
if yaml_p['clear']:
    clear('test')
