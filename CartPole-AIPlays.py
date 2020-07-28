import gym
from numpy import argmax
from tensorflow.keras.models import load_model

env = gym.make('CartPole-v0')
env.reset()
model = load_model('OpenAI-Model.h5')
print(model.summary())

goal_steps = 500
for episode in range(5):
	print(f'Episode : {episode+1}')
	current_condition = env.reset()
	for _ in range(goal_steps):
		env.render(mode='rgb_array')

		action = argmax(model.predict(current_condition.reshape(1,-1)))
		obs,reward,done,info = env.step(action)
		current_condition = obs
		
		if done:
			break
env.close()