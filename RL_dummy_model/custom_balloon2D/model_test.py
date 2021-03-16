from build_environment import balloon2d
from build_agent import DQN_RND
from analysis import plot_reward, plot_path

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
alg = DQN_RND(env)
alg.load_weights('process' + str(yaml_p['process_nr']).zfill(5) + '/weights_model/')

for i in range(15):
    # reset
    obs = env.reset()

    # loop
    while True:
        x = torch.Tensor(obs).unsqueeze(0)
        Q = alg.model(x)
        action = Q.argmax().detach().item()
        new_obs, reward, done, info = env.step(action)
        obs = new_obs
        env.render(mode=True)

        if done:
            break
