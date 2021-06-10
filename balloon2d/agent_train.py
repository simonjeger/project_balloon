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

from build_environment import balloon2d
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

if yaml_p['phase'] < yaml_p['cherry_pick']:
    print('Please choose a higher phase / lower cherr_pick')
if yaml_p['cherry_pick']:
    phase = int(yaml_p['phase']/yaml_p['cherry_pick'])
else:
    phase = int(yaml_p['phase'])

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
for r in range(yaml_p['curriculum_rad']):
    if yaml_p['curriculum_rad'] == 1:
        radius_x = yaml_p['radius_stop_x']
        radius_z = yaml_p['radius_stop_x']
    else:
        x = r/(yaml_p['curriculum_rad']-1)
        radius_x = yaml_p['radius_start_x'] - (yaml_p['radius_start_x'] - yaml_p['radius_stop_x'])*x
        radius_z = radius_x*(1+(1-x)*(yaml_p['radius_start_ratio']-1))

    env = balloon2d(epi_n,step_n,'train',writer,radius_x, radius_z)
    ag = Agent(epi_n,step_n,'train',env,writer)
    if (r > 0) | load_prev_weights:
        ag.load_weights(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/')

    current_phase = [-np.inf]*phase
    best_phase = current_phase[:]
    dry = 0

    while time.time() < init_time + yaml_p['time_train']:
        if yaml_p['cherry_pick'] > 0:
            if epi_n%yaml_p['cherry_pick'] == 0:
                ag.stash_weights()
        else:
            ag.stash_weights()

        log = ag.run_epoch()
        h = int((time.time() - init_time)/3600)
        m = int(((time.time() - init_time) - h*3600)/60)
        s = int((time.time() - init_time) - h*3600 - m*60)
        print('runtime: ' + str(h) + ':' + str(m) + ':' + str(s) + ' epoch: ' + str(epi_n) + ' reward: ' + str(log))

        # save weights
        if yaml_p['cherry_pick'] > 0:
            if epi_n%yaml_p['cherry_pick'] == 0:
                env_val = balloon2d(epi_n,step_n,'val')
                ag_val = Agent(epi_n,step_n,'val',env_val)
                ag_val.load_stash()
                with ag_val.agent.eval_mode():
                    log_val = ag_val.run_epoch()
                current_phase[int((epi_n/yaml_p['cherry_pick'])%phase)] = log_val
                writer.add_scalar('reward_epi_val', log_val, env.step_n-1)
                print('reward_val: ' + str(log_val))

                if sum(current_phase) > sum(best_phase):
                    ag.save_weights(current_phase, yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/')
                    best_phase = current_phase[:]
                    dry = 0

            if (dry > yaml_p['curriculum_rad_dry']) & (yaml_p['curriculum_rad'] + 1 < r):
                dry = 0
                epi_n += 1
                break

        dry += 1
        epi_n += 1
    step_n = env.step_n

    if yaml_p['cherry_pick'] == 0:
        ag.save_weights(current_phase, yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/')

ag.clear_stash()

plot_reward()
plot_path()

# Need to clear out for log files during testing
if yaml_p['clear'] & (not yaml_p['reuse_weights']):
    clear('train')
