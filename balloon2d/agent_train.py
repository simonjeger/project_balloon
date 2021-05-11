from analysis import plot_reward, plot_path, plot_qmap, write_overview, clear
# clear out any previously created log files and data
clear()

from build_environment import balloon2d
from build_agent import Agent

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

from torch.utils.tensorboard import SummaryWriter

# initialize logger
writer = SummaryWriter(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/logger')

# model_train
num_epochs = int(yaml_p['num_epochs']/yaml_p['curriculum_rad'])

# index for log file
duration = yaml_p['duration']
fps = min(int(num_epochs/duration),yaml_p['fps'])
n_f = duration*fps
ratio = num_epochs/n_f

phase = yaml_p['phase']
current_phase = [-np.inf]*phase
best_phase = current_phase[:]

epi_n = 0
step_n = 0

for r in range(yaml_p['curriculum_rad']):
    radius_z = yaml_p['radius_z'] - (yaml_p['radius_z'] - yaml_p['radius_x'])*(r+1)/yaml_p['curriculum_rad']

    env = balloon2d(epi_n,step_n,'train',writer,radius_z)
    ag = Agent(epi_n,step_n,'train',env,writer)
    if r > 0:
        ag.load_weights(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/')

    for i in range(num_epochs):
        ag.stash_weights()
        log = ag.run_epoch(False)
        current_phase[i%phase] = log
        print('epoch: ' + str(i) + ' reward: ' + str(log))

        # save weights
        if yaml_p['cherry_pick']:
            env_val = balloon2d(epi_n,step_n,'val',radius_z=radius_z)
            ag_val = Agent(epi_n,step_n,'val',env_val)
            ag_val.load_stash()
            with ag_val.agent.eval_mode():
                log_val = ag_val.run_epoch(False)
            current_phase[i%phase] = log_val
            writer.add_scalar('reward_epi_val', log_val, env.step_n-1)
            print('epoch: ' + str(i) + ' reward_val: ' + str(log_val))

            if sum(current_phase) > sum(best_phase):
                ag.save_weights(current_phase, yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/')
                best_phase = current_phase[:]


        # write in log file
        if (np.floor(i%ratio) == 0) & yaml_p['qfunction']:
            Path(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/log_qmap/').mkdir(parents=True, exist_ok=True)
            Q_vis = ag.visualize_q_map()
            torch.save(Q_vis, yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/log_qmap/log_qmap_' + str(i).zfill(5) + '.pt')

    if not yaml_p['cherry_pick']:
        ag.save_weights(current_phase, yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/')

    epi_n = env.epi_n
    step_n = env.step_n

ag.clear_stash()

plot_reward()
plot_path()
if yaml_p['qfunction']:
    plot_qmap()

# Need to clear out for log files during testing
clear()
