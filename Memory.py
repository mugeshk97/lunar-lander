import numpy as np

class Memory():
	def __init__(self, max_size, input_dim, n_actions):
		self.mem_size = max_size
		self.mem_count = 0
		self.n_actions = n_actions
		self.state_memory = np.zeros((self.mem_size, *input_dim), dtype=np.float32)
		self.new_state_memory = np.zeros((self.mem_size,  *input_dim), dtype=np.float32)
		self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.uint8)  
		self.reward_memory = np.zeros((self.mem_size), dtype=np.float32)
		self.terminal_memory = np.zeros((self.mem_size), dtype=np.uint8)

	def store_transition(self, state, action, reward, state_, done):
		index = self.mem_count % self.mem_size

		self.state_memory[index] = state
		self.new_state_memory[index] = state_
		actions = np.zeros(self.n_actions)
		actions[action] = 1.0
		self.action_memory[index] = actions
		self.reward_memory[index] = reward
		self.terminal_memory[index] = 1 - done
		self.mem_count += 1

	def sample_memory(self, batch_size):
		max_mem = min(self.mem_count, self.mem_size)

		batch = np.random.choice(max_mem, batch_size, replace=False)

		states = self.state_memory[batch]
		actions = self.action_memory[batch]
		rewards = self.reward_memory[batch]
		states_ = self.new_state_memory[batch]
		terminal = self.terminal_memory[batch]

		return states, actions, rewards, states_, terminal

