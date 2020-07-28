print('Importing Libraries...')

import datetime
import gym
import random
import numpy as np
from tensorflow.keras.layers import Dense,Dropout
# from tflearn.layers.core import input_data,droput,fully_connected
# from tflearn.layers.estimators import regression

print('Creating Environment...')

# Environment
env = gym.make('CartPole-v0')
env.reset()

lr = 0.001
goal_steps = 500
score_requirement = 70
initial_games = 10000

training_data = []
scores = []
acc_scores = []

for episode in range(initial_games):
	score = 0
	game_memory = []
	prev_observation = []
	for _ in range(goal_steps):
		# env.render(mode='rgb_array')
		action = random.randint(0,1)
		obs,reward,done,info = env.step(action)

		if len(prev_observation) > 0:
			game_memory.append([prev_observation,action])

		prev_observation = obs
		score += reward

		if done:
			break

	# Run calculation (once per episode)  when the game done
	if score > score_requirement:
		acc_scores.append(score)
		for data in game_memory:
			# If action == 1
			if data[1] == 1:
				output = [0,1]
			elif data[1] == 0:
				output = [1,0]

			# Append [observation,output]
			training_data.append([data[0],output])

	env.reset()
	scores.append(score)

# ----Outside Loop----#

training_data_saved = np.array(training_data)
np.save(f'training_data_{datetime.datetime.now().strftime("%m%d%Y-%H%M%S")}.npy',training_data_saved)
env.close()

print(f'All accepted score : {len(acc_scores)}')
print(acc_scores)
print(f'Average accepted score : {np.mean(acc_scores)}')
print(f'Median accepted score : {np.median(acc_scores)}')