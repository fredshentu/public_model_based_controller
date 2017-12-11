"""
Author: Dian Chen
Note that this script only applies to Box3dReachPixel environments

python data_collector.py Box3dPush-v0 /home/fredshentu/Desktop/dynamic_model_planner_data/Box3dPush-v0_policy_data --policy_traj ~/Desktop/rllab/data/local/trpo-box3d-state-push-v0-tf/trpo_box3d_state_push_v0_tf_2017_08_02_21_35_12_0001/itr_200.pkl --state_obs 

"""

from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc import logger

import argparse
import time
import joblib
import tensorflow as tf
import numpy as np

NUM_CHUNK=20000


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def get_state(env):
	d = env.wrapped_env._wrapped_env.env.env.model.data
	qpos = d.qpos.flatten()
	qvel = d.qvel.flatten()
	return np.concatenate([qpos, qvel])

# def get_state(env):
# 	d = env.wrapped_env._wrapped_env.env.env.model.data
# 	qpos = d.qpos.flatten()
# 	qvel = d.qvel.flatten()

# 	return np.concatenate([qpos[:4], qvel[:4]])


def main():
	
	parser = argparse.ArgumentParser()
	parser.add_argument('env', type=str,
						help='name of gym env')
	parser.add_argument('name', type=str, 
						help='name of database to store')
	parser.add_argument('--num', type=int, default=300000,
						help='number of samples to collect')
	parser.add_argument('--with_state', action='store_true')
	parser.add_argument('--state_obs', action='store_true')
	parser.add_argument('--start_index', type=int, default=0)
	parser.add_argument('--restore_env', type=str, default=None)
	parser.add_argument('--policy_traj', type=str, default=None)
	args = parser.parse_args()

	# Build env
	env = TfEnv(normalize(env=GymEnv(args.env,record_video=False, \
	log_dir='/tmp/gym_test',record_log=False)))

	if args.restore_env is not None:
		with tf.Session() as sess:
			data = joblib.load(args.restore_env)
			env = data['env']

	with tf.Session() as sess:
		if args.policy_traj is not None:
			data = joblib.load(args.policy_traj)
			env = data['env']
			policy = data['policy']
		for i in range(args.num // NUM_CHUNK):
			filename = args.name + '/' + str(i + args.start_index) + '.tfrecord'
			writer = tf.python_io.TFRecordWriter(filename)
			logger.log('Start collecting data, saving to {}'.format(filename))
	
			obs = env.reset()
			obs = env.reset()
			obs, r, d, env_info = env.step(np.zeros(4))
			transfered_obs = env_info['coordinate_transfered_obs']
			next_obs = None
	
			start_time = time.time()
			j = 0
			while j < NUM_CHUNK:
				if args.policy_traj is not None:
					policy_action,_ = policy.get_action(obs)
					action = np.clip(policy_action, -1, 1)
					# import pdb; pdb.set_trace()
				else:
					action = env.action_space.sample()
				next_obs, reward, done, env_info = env.step(action)
				next_transfered_obs = env_info['coordinate_transfered_obs']
				# env.render()
				if args.state_obs:
					# import pdb; pdb.set_trace()
					feature = {
						'obs': _floats_feature(obs),
						'next_obs': _floats_feature(next_obs),
						'action': _floats_feature(action),
						'relate_obs': _floats_feature(transfered_obs),
						'next_relate_obs':_floats_feature(next_transfered_obs),
					}
				else:
					feature = {
						'obs': _bytes_feature(obs.astype(np.uint8).tostring()),
						'next_obs': _bytes_feature(next_obs.astype(np.uint8).tostring()),
						'action': _floats_feature(action),
					}
	
				if args.with_state:
					state = get_state(env)
					feature['state'] = _floats_feature(state)
				if env_info['contact']:
					j+=1
					# env.render()
					# print("transition involve contact, saving index {}".format(j))
					example = tf.train.Example(features=tf.train.Features(feature=feature))
					
					writer.write(example.SerializeToString())
	
				if done:
					obs = env.reset()
					obs, r, d, env_info = env.step(np.zeros(4))
					transfered_obs = env_info['coordinate_transfered_obs']
				else:
					obs = next_obs
					transfered_obs = next_transfered_obs
			writer.close()
	
			logger.log('Finished collecting, elapsed time: {}'.format(time.time() - start_time))


if __name__ == "__main__":
	main()