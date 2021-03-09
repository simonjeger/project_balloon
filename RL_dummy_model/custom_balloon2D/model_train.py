from build_model import build_model
from build_agent import build_agent
from build_environment import balloon2d

import gym
import random
import numpy as np
from tensorflow.keras.optimizers import Adam

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
dqn = build_agent(model, actions,'train')
dqn.compile(Adam(lr=1e-3), metrics=['mae']) #lr is learning rate
dqn.fit(env, nb_steps=3000000, visualize=False, verbose=1) #verbose is just an option on how to display the fitting process

# save agent / trained weights
dqn.save_weights('weights_model/dqn_weights.h5f', overwrite=True)
