from rl.agents import DQNAgent #deep Q-network (DQN) algorithm is a model-free, online, off-policy reinforcement learning method
from rl.policy import BoltzmannQPolicy
from rl.policy import MaxBoltzmannQPolicy
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

def build_agent(model, actions):
    policy = BoltzmannQPolicy() #instead of always taking a random or optimal action, this approach involves choosing an action with weighted probabilitie
    policy = MaxBoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1) #we store #window_length of windows for #limit of episodes
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2) #nb_steps_warmup where agent collects information before training (doesn't learn in the first #nb_steps_warmup)
    return dqn
