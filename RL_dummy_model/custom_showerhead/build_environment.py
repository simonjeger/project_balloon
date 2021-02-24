from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random

class ShowerEnv(Env):
    def __init__(self):
        # action we can take: down, stay, up
        self.action_space = Discrete(3) #discrete space
        # temperature array
        self.observation_space = Box(low=np.array([0]), high=np.array([100])) #continious space
        # set start temp
        self.state = 38 + random.randint(-3, 3)
        # set shower length
        self.shower_length = 60

    def step(self, action):
        # apply action
        self.state += action -1
        # reduce shower length by 1 second
        self.shower_length -= 1

        # calculate reward
        if self.state >= 37 and self.state <= 39:
            reward = 1
        else:
            reward = -1

        # check if shower is done
        if self.shower_length <= 0:
            done = True
        else:
            done = False

        # apply temperature noise
        self.state += random.randint(-1,1)
        # set placeholder for info
        info = {}

        # return step information
        return self.state, reward, done, info

    def render(self):
        # implement vizualisation
        pass

    def reset(self):
        # reset shower temperature
        self.state = 38 + random.randint(-3,3)
        # reset shower time
        self.shower_length = 60
        return self.state
