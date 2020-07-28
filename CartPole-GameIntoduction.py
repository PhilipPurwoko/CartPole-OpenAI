import gym

# Environment
env = gym.make('CartPole-v0')
env.reset()

goal_steps = 500
for episode in range(5):
	env.reset()
	for _ in range(goal_steps):
		env.render(mode='rgb_array')

		action = env.action_space.sample()
		obs,reward,done,info = env.step(action)
		if done:
			break
env.close()