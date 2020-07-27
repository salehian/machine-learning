import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import gym

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.distributions import Categorical


env = gym.make('CartPole-v0')
env.seed(0)
print('observation space:', env.observation_space)
print('action space:', env.action_space)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")