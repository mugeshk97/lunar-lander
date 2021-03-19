import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as Optim
import numpy as np


class DQNetwork(nn.Module):
	def __init__(self, learning_rate, input_dim, n_actions):
		super(DQNetwork, self).__init__()
		self.input_dim = input_dim
		self.n_actions = n_actions

		self.fc1 = nn.Linear(*self.input_dim, 256)
		self.fc2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, self.n_actions)
		self.optimizer = Optim.Adam(self.parameters(), lr=learning_rate)
		self.loss = nn.MSELoss()
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
		self.to(self.device)
	
	def forward(self, observation):
		state = T.Tensor(observation).to(self.device)
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		action = self.fc3(x)
		return action


