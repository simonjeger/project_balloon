from analysis import plot_reward, plot_path, write_overview, clear
from build_environment import balloon3d
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

N = 100
f = 0
sucess = []
score = []

while True:
    env = balloon3d(0,0,'test',writer)
    ag = Agent(0,0,'test',env,writer)
    ag.load_weights(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/')
    with ag.agent.eval_mode():
        log_tot = 0
        for i in range(N):
            if f == 0:
                log = ag.run_epoch()
            else:
                log = ag.run_epoch(importance=f-1)
            log_tot += log
            print('analysis: ' + str(int((f+i/N)*100/(env.observation_space.shape[0]+1))) + '% epoch: ' + str(i) + ' reward: ' + str(log))
        sucess.append(env.success_n/N)
        score.append(log_tot/N)
    f += 1
    if f >= env.observation_space.shape[0]+1:
        break

# plot results
x = np.arange(1+env.observation_space.shape[0])
plt.scatter(x,sucess)
plt.scatter(x,score)
plt.legend(['sucess', 'score'])
plt.savefig(yaml_p['process_path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/feature_importance.pdf')

# Delete log files
if yaml_p['clear']:
    clear('test')
