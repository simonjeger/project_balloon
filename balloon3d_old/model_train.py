from build_model import build_model
from build_agent import build_agent
from build_environment import balloon3d

import gym
import random
import numpy as np
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# initialize environment, states and actions
env = balloon3d('train')
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
history = dqn.fit(env, nb_steps=10000, visualize=False, verbose=1) #verbose is just an option on how to display the fitting process

# save agent / trained weights
dqn.save_weights('weights_model/dqn_weights.h5f', overwrite=True)

episode_reward = history.history['episode_reward']
N = int(len(episode_reward)/10)
cumsum = np.cumsum(np.insert(episode_reward, 0, 0))
mean_reward = (cumsum[N:] - cumsum[:-N]) / float(N)

episode_steps = history.history['nb_episode_steps']
cumsum = np.cumsum(np.insert(episode_steps, 0, 0))
mean_steps = (cumsum[N:] - cumsum[:-N]) / float(N)

fig, ax1 = plt.subplots()
ax1.set_xlabel('epoch')
ax1.set_ylabel('reward')
ax1.tick_params(axis='y')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('episode steps')  # we already handled the x-label with ax1
ax2.tick_params(axis='y')

ax2.plot(episode_steps, color='lightblue', alpha=0.75)
ax1.plot(episode_reward, color='orange')
ax2.plot(mean_steps, color='dodgerblue', alpha=0.75)
ax1.plot(mean_reward, color='firebrick')

fig.tight_layout()  # otherwise the right y-label is slightly clipped

ax1.legend(['reward per episode', 'mean reward'], loc='upper left')
ax2.legend(['steps per episode', 'mean steps'], loc='upper right')
plt.show()
plt.close()
