from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv

from railrl.predictors.dynamics_model import FullyConnectedEncoder

import numpy as np
import tensorflow as tf
import itertools
from os import listdir
import os.path as osp

import joblib
import argparse


import matplotlib.pyplot as plt


ACTION_DIM_MAP = {
	"Box3dReach-v12": 12,
	"Box3dReachPixel-v11" : 4,
	"test-fred": 4,
}

OBS_SHAPE_MAP = {
	"Box3dReach-v12": [86],
	"Box3dReachPixel-v11" : [84, 84, 8],
	"test-fred": [84, 84, 4],
}

BOXES_POS_INDEX = {
	"Box3dReach-v12": [4,5,6,7,11,12,13,14,18,19,20,21,25,26,27,28,32,33,34,35,39,40,41,42],
}


def set_state(env, state, qpos_dim):
	inner_env = env.wrapped_env._wrapped_env.env.env
	inner_env.set_state(state[:qpos_dim], state[qpos_dim:])

def get_qpos(env):
	inner_env = env.wrapped_env._wrapped_env.env.env
	return inner_env.model.data.qpos.flatten()

def get_qvel(env):
	inner_env = env.wrapped_env._wrapped_env.env.env
	return inner_env.model.data.qvel.flatten()

def get_render_img(env):
	return env.wrapped_env._wrapped_env.env.env.render(mode='rgb_array')

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('file', type=str, default='path to snapshot file')
	parser.add_argument('--env_name', type=str, default='Box3dReach-v12')

	args = parser.parse_args()

	with tf.Session() as sess:
		data = joblib.load(args.file)
		env = data['env']

		s_ph = tf.placeholder(tf.float32, [None] + OBS_SHAPE_MAP[args.env_name])

		_encoder = data['encoder']
		_regressor = data['regressor']

		encoder = _encoder.get_weight_tied_copy(observation_input=s_ph)
		regressor = _regressor.get_weight_tied_copy(observation_input=encoder.output)

		ob = env.reset()
		qpos_dim = env.wrapped_env._wrapped_env.env.env.init_qpos.shape[0]

		plt.ion()
		f, [ax1, ax2] = plt.subplots(1, 2)
		ax1.set_title("Real")
		ax2.set_title("Box pos inferred from feature")

		while True:
			old_state = np.concatenate([get_qpos(env), get_qvel(env)])
			img1 = get_render_img(env)
			pred_loc = sess.run(regressor.output, 
				{ s_ph: [ob] }
			)

			qpos = get_qpos(env)
			qvel = get_qvel(env)

			qpos[BOXES_POS_INDEX[args.env_name]] = pred_loc

			set_state(env, np.concatenate([qpos, qvel]), qpos_dim)
			img2 = get_render_img(env)
			set_state(env, old_state, qpos_dim)

			ax1.imshow(img1)
			ax2.imshow(img2)
			plt.show()
			plt.pause(1.0)

			ob, _, _, _ = env.step(env.action_space.sample())


if __name__ == '__main__':
	main()			