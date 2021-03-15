from build_model import build_model
from build_agent import build_agent

import gym
import random
import numpy as np
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# initialize environment, states and actions
env = gym.make('CartPole-v0')
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
        env.render()
        action = random.choice([0,1])
        n_state, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episodes, score))
"""

# create a deep learning model with Keras
model = build_model(states, actions)

# build agent with Keras-RL
dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
history = dqn.fit(env, nb_steps=200000, visualize=False, verbose=1)

# save agent / trained weights
dqn.save_weights('weights/dqn_weights.h5f', overwrite=True)

# plot
episode_reward = history.history['episode_reward']
N = int(len(episode_reward)/10)
cumsum = np.cumsum(np.insert(episode_reward, 0, 0))
mean_reward = (cumsum[N:] - cumsum[:-N]) / float(N)

plt.plot(episode_reward)
plt.plot(mean_reward)

plt.ylabel('reward')
plt.xlabel('epoch')
plt.legend(['reward per episode', 'mean reward'], loc='upper left')
plt.show()
plt.close()
