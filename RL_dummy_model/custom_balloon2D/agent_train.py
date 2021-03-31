from build_environment import balloon2d
from build_agent import Agent
from analysis import plot_reward, plot_path, plot_qmap, write_overview, clear

import numpy as np
import gym
import matplotlib.pyplot as plt
import torch
from pathlib import Path

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

env = balloon2d('train')
ag = Agent(env)

# clear out any previously created log files and data
clear()

# model_train
num_epochs = yaml_p['num_epochs']

# index for log file
duration = yaml_p['duration']
fps = min(int(num_epochs/duration),yaml_p['fps'])
n_f = duration*fps
ratio = num_epochs/n_f

phase = yaml_p['phase']
current_phase = [-np.inf]*phase
best_phase = current_phase[:]

for i in range(num_epochs):
    ag.stash_weights()
    log = ag.run_epoch(False)
    current_phase[i%phase] = log
    print('epoch: ' + str(i) + ' reward: ' + str(log))

    # save weights if improved
    if sum(current_phase) > sum(best_phase):
        ag.save_weights(current_phase, 'process' + str(yaml_p['process_nr']).zfill(5) + '/')
        best_phase = current_phase[:]

    # write in log file
    if np.floor(i%ratio) == 0:
        Path('process' + str(yaml_p['process_nr']).zfill(5) + '/log_qmap/').mkdir(parents=True, exist_ok=True)
        Q_vis = ag.visualize_q_map()
        torch.save(Q_vis, 'process' + str(yaml_p['process_nr']).zfill(5) + '/log_qmap/log_qmap_' + str(i).zfill(5) + '.pt')

ag.clear_stash()

# analyse
plot_reward()
plot_path()
plot_qmap()

if yaml_p['overview']:
    write_overview()

# Delete log files
if yaml_p['clear']:
    clear()