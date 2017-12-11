import gym
import matplotlib.pyplot as plt
import scipy.misc as sci

env = gym.make('Box3dReachPixel-v0')
env.reset()
for t in range(100):
    action = env.action_space.sample()
    obs, _, _, _ = env.step(action)
    sci.imsave("{}.png".format(t), obs[:,:,0])
    
