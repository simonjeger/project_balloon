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

env = balloon2d('train')
ag = Agent(env)
ag.load_weights(yaml_p['path'] + 'process' + str(yaml_p['process_nr']).zfill(5) + '/')

with ag.agent.eval_mode():
    for i in range(15):
        log = ag.run_epoch(True)
        print('epoch: ' + str(i) + ' reward: ' + str(log))
