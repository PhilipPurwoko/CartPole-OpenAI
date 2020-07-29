print('Importing Libraries...')

import time
import os
import datetime
import gym
import random
import numpy as np
from numpy import argmax
# from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
# import matplotlib.pyplot as plt

print('Creating Environment...')

# Environment
env = gym.make('CartPole-v0')
env.reset()
# model = load_model('OpenAI-Model.h5')

goal_steps = 500
score_requirement = 60
initial_games = 10000

def initial_population():
	training_data = []
	x_train = []
	y_train = []
	scores = []
	acc_scores = []

	print('Running observation...')
	print('(May take few mintues)')
	for episode in range(initial_games):
		score = 0
		game_memory = []
		prev_observation = env.reset()
		for _ in range(goal_steps):
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

				x_train.append(data[0])
				y_train.append(output)

		env.reset()
		scores.append(score)

	# ----Outside Loop----#

	env.close()

	print(f'All accepted score : {len(acc_scores)}')
	print(f'Average accepted score : {np.mean(acc_scores)}')
	print(f'Median accepted score : {np.median(acc_scores)}')

	return (x_train,y_train)

def neural_network_model():
	model = Sequential([
		Flatten(input_shape=[4,]),
		Dense(128,activation='relu'),
		Dropout(0.5),
		Dense(256,activation='relu'),
		Dropout(0.5),
		Dense(512,activation='relu'),
		Dropout(0.5),
		Dense(256,activation='relu'),
		Dropout(0.5),
		Dense(128,activation='relu'),
		Dropout(0.5),
		Dense(2,activation='softmax')
	])

	model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

	return model

def train_model(training_data,model=False):
	x_train = np.array(training_data[0])
	y_train = np.array(training_data[1])

	if not model:
		model = neural_network_model()

	def createCallback():
		os.system('load_ext tensorboard')
		os.makedirs('logs',exist_ok=True)
		logdir = os.path.join('logs',datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
		return TensorBoard(logdir)
	
	calback = createCallback()
	earlyStoping = EarlyStopping(monitor='loss',patience=3)

	model.fit(x_train,y_train,batch_size=32,epochs=100,verbose=1,callbacks=[calback,earlyStoping])
	return model

def ai_plays(model):
	scores = []
	choices = []
	survival_scores = []
	for game in range(10):
		time_start = time.time()
		print(f'Game ({game+1} of 10)')
		score = 0
		game_memory = []
		prev_obs = env.reset()
		for _ in range(goal_steps):
			env.render()
			if len(prev_obs) == 0:
				action = random.randint(0,1)
			else:
				action = argmax(model.predict(prev_obs.reshape(1,-1)))
			choices.append(action)

			new_obs, reward, done, info = env.step(action)
			prev_obs = new_obs
			game_memory.append([new_obs,action])
			score += reward
			if done:
				break
		scores.append(score)
		time_end = time.time()
		survival_time = time_end - time_start
		survival_scores.append(survival_time)
		print(f'Survive time : {survival_time} seconds')
	print(f'Best survival time : {np.max(survival_scores)}')

	print(f'Average accepted score : {np.mean(scores)}')
	print(f'Median accepted score : {np.median(scores)}')

if __name__ == '__main__':
	nn_model = neural_network_model()
	training_data = initial_population()

	print(np.array(training_data[0]).shape)
	print(np.array(training_data[1]).shape)

	model = train_model(training_data,nn_model)
	ai_plays(model)