from analysis import plot_reward, plot_path, write_overview, clear

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

# clear out any previously created log files and data
if not yaml_p['reuse_weights']:
    clear('train')
    clear('test')

from build_environment import balloon3d
from build_agent import Agent
from utils.load_tf import tflog2pandas, many_logs2pandas

import numpy as np
import gym
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import os
import time

from torch.utils.tensorboard import SummaryWriter

# initialize logger
writer = SummaryWriter(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_train')

# model_train
init_time = time.time()

# Index epi_n and step_n
path_logger = yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger_train/'
name_list = os.listdir(path_logger)
for i in range(len(name_list)):
    name_list[i] = path_logger + name_list[i]
df = many_logs2pandas(name_list)

if len(df) > 0:
    epi_n = df['epi_n'].dropna().iloc[-1] + 1
    step_n = df['step_n'].dropna().iloc[-1] + 1
    load_prev_weights = True
else:
    epi_n = 0
    step_n = 0
    load_prev_weights = False

# training process
env = balloon3d(epi_n,step_n,'train',writer)
ag = Agent(epi_n,step_n,'train',env,writer)
if load_prev_weights:
    ag.load_weights(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/')

while time.time() < init_time + yaml_p['time_train']:
    log = ag.run_epoch()
    h = int((time.time() - init_time)/3600)
    m = int(((time.time() - init_time) - h*3600)/60)
    s = int((time.time() - init_time) - h*3600 - m*60)
    print('runtime: ' + str(h) + ':' + str(m) + ':' + str(s) + ' epoch: ' + str(int(epi_n)) + ' reward: ' + str(log))

    # save weight as a backup every N episodes
    if epi_n%10 == 0:
        ag.save_weights(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/')

    epi_n += 1

# save weights
ag.save_weights(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/')

step_n = env.step_n

plot_reward()
plot_path()
