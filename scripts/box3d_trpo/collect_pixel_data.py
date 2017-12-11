from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.normalized_env import normalize
from railrl.data_management.simple_replay_pool import SimpleReplayPool
from rllab.envs.gym_env import GymEnv

import gym
import numpy as np
import pickle

TOTAL_NUM = 2000000
NUM = 100000

env = TfEnv(normalize(env=GymEnv('Box3dReachPixel-v1',record_video=False, \
    log_dir='/tmp/gym_test',record_log=False)))

obs_shape = env.spec.observation_space.shape
state_shape = env.wrapped_env.wrapped_env.env.env.init_qpos.shape
action_shape = env.spec.action_space.shape

obs = env.reset()
state = env.wrapped_env.wrapped_env.env.env.model.data.qpos

obs_list = np.zeros([NUM] + list(obs_shape), np.uint8)
state_list = np.zeros([NUM] + list(state_shape), np.float32)
action_list = np.zeros([NUM] + list(action_shape), np.float32)
done_list = np.zeros([NUM],  np.uint8)
term_list = np.zeros([NUM], np.uint8)

j = 0
try: 
	obs_list = np.zeros([NUM] + list(obs_shape), np.uint8)
	state_list = np.zeros([NUM] + list(state_shape), np.float32)
	action_list = np.zeros([NUM] + list(action_shape), np.float32)
	done_list = np.zeros([NUM],  np.uint8)
	term_list = np.zeros([NUM], np.uint8)

	for j in range(TOTAL_NUM / NUM):
		i = 0
		while i < NUM:
			if i % 10000 == 0:
				print ("Collected: %d samples"%i)
			action = env.action_space.sample()
			next_obs, r, done, _ = env.step(action)

			obs_list[i] = obs
			action_list[i] = action
			done_list[i] = done
			term_list[i] = False

			if done:
				obs = env.reset()
			else:
				next_obs = obs
			i += 1

		save_dict = { "obs":obs_list,"action_list":action_list,"done_list":done_list,"term_list":term_list, "state_list":state_list }
		with open("/home/fred/pixel-data/{}.pkl".format(j), 'wb+') as handle:
			pickle.dump(save_dict, handle)

except KeyboardInterrupt:
	print ("Terminated. Collected %d samples" % (NUM+i))
	pass