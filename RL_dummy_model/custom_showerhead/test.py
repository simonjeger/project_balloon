from build_model import build_model
from build_agent import build_agent
from build_environment import ShowerEnv

import gym
import numpy as np
from tensorflow.keras.optimizers import Adam

# initialize model again and load weights
env = ShowerEnv()
actions = env.action_space.n
states = env.observation_space.shape
model = build_model(states, actions)
dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.load_weights('weights/dqn_weights.h5f')

# show result and print scores
scores = dqn.test(env, nb_episodes=15, visualize=False)
print(np.mean(scores.history['episode_reward']))
