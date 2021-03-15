from rl.agents import DQNAgent #deep Q-network (DQN) algorithm is a model-free, online, off-policy reinforcement learning method
from rl.policy import LinearAnnealedPolicy, SoftmaxPolicy, EpsGreedyQPolicy, GreedyQPolicy, BoltzmannQPolicy, MaxBoltzmannQPolicy, BoltzmannGumbelQPolicy
from rl.memory import SequentialMemory

import yaml
import argparse

# Get yaml parameter
parser = argparse.ArgumentParser()
parser.add_argument('yaml_file')
args = parser.parse_args()
with open(args.yaml_file, 'rt') as fh:
    yaml_p = yaml.safe_load(fh)

def build_agent(model, actions, train_or_test):
    if train_or_test == 'train':
        policy = MaxBoltzmannQPolicy(eps=yaml_p['eps'])
    elif train_or_test == 'test':
        policy = MaxBoltzmannQPolicy(eps=0)
    else:
        print('do you want to train or test?')
    memory = SequentialMemory(limit=50000, window_length=1) #we store #window_length of windows for #limit of episodes
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=100, target_model_update=yaml_p['target_model_update']) #nb_steps_warmup where agent collects information before training (doesn't learn in the first #nb_steps_warmup)
    return dqn

"""
LinearAnnealedPolicy()
Linear Annealing Policy computes a current threshold value and
    transfers it to an inner policy which chooses the action. The threshold
    value is following a linear function decreasing over time.

SoftmaxPolicy()
Implement softmax policy for multinimial distribution. Simple Policy that takes action according to the pobability distribution

EpsGreedyQPolicy()
Eps Greedy policy either:
    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)

GreedyQPolicy()
Implement the greedy policy. Greedy policy returns the current best action according to q_values

BoltzmannQPolicy()
Policy Boltzmann Q Policy builds a probability law on q values and returns an action selected randomly according to this law.

MaxBoltzmannQPolicy()
A combination of the eps-greedy and Boltzman q-policy. Wiering, M.: Explorations in Efficient Reinforcement Learning. PhD thesis, University of Amsterdam, Amsterdam (1999) https://pure.uva.nl/ws/files/3153478/8461_UBA003000033.pdf

BoltzmannGumbelQPolicy()
Implements Boltzmann-Gumbel exploration (BGE) adapted for Q learning based on the paper Boltzmann Exploration Done Right (https://arxiv.org/pdf/1705.10257.pdf).
BGE is invariant with respect to the mean of the rewards but not their variance. The parameter C, which defaults to 1, can be used to correct for this, and should be set to the least upper bound on the standard deviation of the rewards.
BGE is only available for training, not testing. For testing purposes, you can achieve approximately the same result as BGE after training for N steps on K actions with parameter C by using the BoltzmannQPolicy and setting tau = C/sqrt(N/K).
"""
