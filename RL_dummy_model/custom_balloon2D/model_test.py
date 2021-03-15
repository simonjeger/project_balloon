from build_model import build_model
from build_agent import build_agent
from build_environment import balloon2d

import gym
import numpy as np
from tensorflow.keras.optimizers import Adam

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

# initialize model again and load weights
env = balloon2d('test')
actions = env.action_space.n
states = env.observation_space.shape[0]
model = build_model(states, actions)
dqn = build_agent(model, actions,'test')
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.load_weights('process' + str(yaml_p['process_nr']).zfill(5) + '/weights_model/dqn_weights.h5f')

# show result and print scores
scores = dqn.test(env, nb_episodes=15, visualize=True)
print(np.mean(scores.history['episode_reward']))
