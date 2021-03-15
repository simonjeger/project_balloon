from build_model import build_model
from build_agent import build_agent
from build_environment import balloon2d
from analysis import plot_reward, plot_path

import gym
import random
from tensorflow.keras.optimizers import Adam

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

# initialize environment, states and actions
env = balloon2d('train')
states = env.observation_space.shape[0]
actions = env.action_space.n

# test random environment with OpenAI Gym
"""
episodes = 10
for episodes in range(1,episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        #env.render()
        action = env.action_space.sample())
        n_state, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episodes, score))
"""

# create a deep learning model with Keras
model = build_model(states, actions)

# build agent with Keras-RL
dqn = build_agent(model, actions, 'train')
dqn.compile(optimizer=Adam(lr=1e-3), metrics=['mae']) #lr is learning rate (used to be 1e-3)
history = dqn.fit(env, nb_steps=yaml_p['nb_steps'], visualize=False, verbose=1) #verbose is just an option on how to display the fitting process

# save agent / trained weights
dqn.save_weights('process' + str(yaml_p['process_nr']).zfill(5) + '/weights_model/dqn_weights.h5f', overwrite=True)

# post analysis
plot_reward(history)
plot_path()
