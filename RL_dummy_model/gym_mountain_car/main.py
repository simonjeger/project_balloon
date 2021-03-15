import numpy as np
import gym
from build_agent_2 import DQN_RND
import matplotlib.pyplot as plt
import torch

from build_environment import balloon2d
from analysis import plot_reward, plot_path

env = balloon2d('train')

gamma = 0.95
alg = DQN_RND(env,gamma,10000)

# model_train
num_epochs = 50
for i in range(num_epochs):
    log = alg.run_epoch()
    print('epoch: {}. return: {}'.format(i,np.round(log.get_current('real_return')),2))

# model_test
obs = env.reset()
for t in range(1000):
    x = torch.Tensor(obs).unsqueeze(0)
    Q = alg.model(x)
    action = Q.argmax().detach().item()
    new_obs, reward, done, info = env.step(action)
    obs = new_obs
    env.render(mode=True)

    if done:
        break

# analyse
plot_reward()
plot_path()
