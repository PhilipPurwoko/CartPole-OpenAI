print('Importing Libraries...')

import datetime
import gym
import random
import numpy as np

print('Creating Environment...')

# Environment
env = gym.make('CartPole-v0')
env.reset()

goal_steps = 1000
score_requirement = 85
initial_games = 100000

training_data = []
x_train = []
y_train = []
scores = []
acc_scores = []

print('Running observation...')
print('(May take few minues)')
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
			# training_data.append((data[0],output))
			x_train.append(data[0])
			y_train.append(output)

	env.reset()
	scores.append(score)

# ----Outside Loop----#

# filename = f'training_data_{datetime.datetime.now().strftime("%m%d%Y-%H%M%S")}.npy'
# filename = 'training_data.npy'
with open('x_train.npy','wb') as file:
	np.save(file,x_train)
with open('y_train.npy','wb') as file:
	np.save(file,y_train)

env.close()

print(f'All accepted score : {len(acc_scores)}')
print(acc_scores)
print(f'Average accepted score : {np.mean(acc_scores)}')
print(f'Median accepted score : {np.median(acc_scores)}')