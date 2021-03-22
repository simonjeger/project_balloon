from build_environment import balloon2d
from build_agent import Agent
from analysis import plot_reward, plot_path, plot_qmap, clear

import numpy as np
import gym
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import shutil

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

# model_train
num_epochs = yaml_p['num_epochs']

# index for log file
duration = yaml_p['duration']
fps = min(int(num_epochs/duration),yaml_p['fps'])
n_f = duration*fps
ratio = num_epochs/n_f
print(duration)
print(fps)
print(n_f)
print(num_epochs)
print(ratio)

for i in range(num_epochs):
    print('epoch: ' + str(i))
    log = ag.run_epoch(False)

    # write in log file
    if np.floor(i%ratio) == 0:
        Path('process' + str(yaml_p['process_nr']).zfill(5) + '/log_qmap/').mkdir(parents=True, exist_ok=True)
        Q_vis = ag.visualize_q_map()
        torch.save(Q_vis, 'process' + str(yaml_p['process_nr']).zfill(5) + '/log_qmap/log_qmap_' + str(i).zfill(5) + '.pt')

        # save weights every now and then as a backup
        ag.save_weights('process' + str(yaml_p['process_nr']).zfill(5) + '/')

# save weights at the end
ag.save_weights('process' + str(yaml_p['process_nr']).zfill(5) + '/')

# analyse
plot_reward()
plot_path()
plot_qmap()
clear() #only clears out log files if yaml parameter is set
