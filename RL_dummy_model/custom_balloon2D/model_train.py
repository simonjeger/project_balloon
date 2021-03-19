from build_environment import balloon2d
from build_agent import DQN_RND
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
alg = DQN_RND(env)

# model_train
num_epochs = yaml_p['num_epochs']
for i in range(num_epochs):
    log = alg.run_epoch()

    print('epoch: {}. return: {}'.format(i,np.round(log.get_current('real_return'),3),2))

    # write in log file
    Path('process' + str(yaml_p['process_nr']).zfill(5) + '/log_qmap/').mkdir(parents=True, exist_ok=True)
    Q_vis = alg.visualize_q_map()
    torch.save(Q_vis, 'process' + str(yaml_p['process_nr']).zfill(5) + '/log_qmap/log_qmap_' + str(i).zfill(5) + '.pt')

alg.save_weights('process' + str(yaml_p['process_nr']).zfill(5) + '/weights_model/')

# analyse
plot_reward()
plot_path()
plot_qmap()
clear() #only clears out log files if yaml parameter is set
