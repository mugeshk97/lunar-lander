import numpy as np
import torch as T
from DQNet import DQNetwork
from Memory import Memory


class Agent():
	def __init__(self, gamma, n_actions, input_dim, mem_size,  epsilon = 1.0, learning_rate = 1e-4, batch_size = 64, epsilon_min = 0.01, epsilon_dec = 1e-5):
		self.gamma = gamma
		self.epsilon = epsilon
		self.n_actions = n_actions
		self.action_space = [i for i in range(n_actions)]
		self.input_dim = input_dim
		self.mem_size = mem_size
		self.batch_size = batch_size
		self.epsilon_min = epsilon_min
		self.epsilon_dec = epsilon_dec

		self.memory = Memory(max_size = self.mem_size, input_dim = self.input_dim, n_actions = self.n_actions)
		self.q_eval = DQNetwork(learning_rate = learning_rate, input_dim = self.input_dim, n_actions = self.n_actions)

	def choose_action(self, observation):
		rand = np.random.random()
		actions = self.q_eval.forward(observation)
		if rand > self.epsilon:
			action = T.argmax(actions).item()
		else:
			action = np.random.choice(self.action_space)
		return action

	def store_transition(self, state, action, reward, state_, done):
		self.memory.store_transition(state, action, reward, state_, done)

	def decrement_epsilon(self):
		self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

	def learn(self):
		if self.memory.mem_count < self.batch_size:
			 return
		self.q_eval.optimizer.zero_grad()

		state, action, reward, new_state, done = self.memory.sample_memory(self.batch_size)
		reward = T.Tensor(reward).to(self.q_eval.device)
		done = T.Tensor(done).to(self.q_eval.device)
		
		action_values = np.array(self.action_space, dtype=np.int32)		
		action_indices = np.dot(action, action_values)	
		batch_index = np.arange(self.batch_size, dtype=np.int32)			

		q_pred = self.q_eval.forward(state).to(self.q_eval.device)
		q_pred_next = self.q_eval.forward(new_state).to(self.q_eval.device)
		q_target = q_pred.clone()

		
		q_target[batch_index, action_indices] = reward + self.gamma * T.max(q_pred_next, dim=1)[0] * done

		self.decrement_epsilon()

		loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
		loss.backward()
		self.q_eval.optimizer.step()
