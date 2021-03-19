from DQAgent import Agent
import gym
import numpy as np


env = gym.make('LunarLander-v2')
agent = Agent(gamma = 0.99, input_dim = [8], n_actions = 4, mem_size = 100000, batch_size = 64)
scores = []
epsilon_history = []
num_game = 10000

score = 0

for i in range(num_game):
	print(f'episode: {i} score: {score} memory: {agent.memory.mem_count} epsilon: {agent.epsilon:3f}')
	epsilon_history.append(agent.epsilon)
	done = False
	observation = env.reset()
	score = 0
	while not done:
		env.render()
		action = agent.choose_action(observation)
		observation_, reward, done, info = env.step(action)
		score += reward
		agent.store_transition(observation, action, reward, observation_, done)
		observation = observation_
		agent.learn()
	scores.append(score)

env.close()