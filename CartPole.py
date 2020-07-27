import gym

# Environment
env = gym.make('CartPole-v0')

# Obsevation
obs = env.reset()

print('Press Ctrl+C to exit...')
while True:
	env.render()

	cart_pos,cart_vel,ang,ang_vel = obs

	if ang > 0:
		action = 1
	else:
		action = 0

	obs,reward,done,info = env.step(action)