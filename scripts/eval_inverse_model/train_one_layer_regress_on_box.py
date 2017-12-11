from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv

from railrl.predictors.dynamics_model import FullyConnectedEncoder, OneLayer

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

def read_and_decode(filename_queue, obs_shape, joint_pos_index):
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(
		  serialized_example,
		  # Defaults are not specified since both keys are required.
		  features={
				  'obs': tf.FixedLenFeature(obs_shape, tf.float32),
				  'next_obs': tf.FixedLenFeature(obs_shape, tf.float32),
				  'action': tf.FixedLenFeature([4], tf.float32),
				  'state': tf.FixedLenFeature([86], tf.float32),
		  })


	# Convert label from a scalar uint8 tensor to an int32 scalar.
	obs = tf.cast(features['obs'], tf.float32)
	next_obs = tf.cast(features['next_obs'], tf.float32)
	action = tf.cast(features['action'], tf.float32)
	state = tf.cast(features['state'], tf.float32)

	joint_pos = tf.gather(state, joint_pos_index)
	
	return obs, next_obs, action, joint_pos


def inputs(filenames, obs_shape, joint_pos_index, train=True, batch_size=128, num_epochs=None):
	"""Reads input data num_epochs times.
	Args:
		train: Selects between the training (True) and validation (False) data.
		batch_size: Number of examples per returned batch.
		num_epochs: Number of times to read the input data, or 0/None to
		   train forever.
	Returns:
		A tuple (images, labels), where:
		* images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
		  in the range [-0.5, 0.5].
		* labels is an int32 tensor with shape [batch_size] with the true label,
		  a number in the range [0, mnist.NUM_CLASSES).
		Note that an tf.train.QueueRunner is added to the graph, which
		must be run using e.g. tf.train.start_queue_runners().
	"""

	with tf.name_scope('input'):
		filename_queue = tf.train.string_input_producer(
				filenames, num_epochs=num_epochs)
				
		image, next_image, action, joint_pos = read_and_decode(filename_queue, obs_shape, joint_pos_index)

		if train:
			num_thread = 12
			queue_capacity = 40000
		else:
			num_thread = 4
			queue_capacity = 4000
		images, next_images, actions, joint_poses = tf.train.batch([image, next_image, action, joint_pos],\
										batch_size = batch_size, num_threads = num_thread,\
										capacity = queue_capacity, enqueue_many =False)

		
		return images, next_images, actions, joint_poses


# BOX_POS_DIM = 42 # Note that this is specific to Box3dReach-v16
BOX_POS_DIM = 24

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('file', type=str, default='path to snapshot file')
	parser.add_argument('output_file', type=str, default='path to save trained model')
	parser.add_argument('--env_name', type=str, default='Box3dReach-v12')
	parser.add_argument('--dataset_path', type=str, default='/media/icm_data/state-v12')
	parser.add_argument('--val_ratio', type=float, default=0.2,
						help="ratio of validation sets")
	parser.add_argument('--render', action='store_true')

	args = parser.parse_args()

	with tf.Session() as sess:
		data = joblib.load(args.file)
		env = data['env']
		# Get dataset
		dataset_names = list(map(lambda file_name: osp.join(args.dataset_path, file_name), listdir(args.dataset_path)))
		val_set_names = dataset_names[:int(len(dataset_names)*args.val_ratio)]
		train_set_names = dataset_names[int(len(dataset_names)*args.val_ratio):]

		train_queue = tf.train.string_input_producer(train_set_names, num_epochs=None)
		val_queue = tf.train.string_input_producer(val_set_names, num_epochs=None)

		obs_shape = OBS_SHAPE_MAP[args.env_name]
		action_dim = ACTION_DIM_MAP[args.env_name]
		train_obs, train_next_obs, train_action, train_joint = inputs(train_set_names, obs_shape, joint_pos_index=BOXES_POS_INDEX[args.env_name], train=True)
		val_obs, val_next_obs, val_action, val_joint = inputs(val_set_names, obs_shape, joint_pos_index=BOXES_POS_INDEX[args.env_name], train=False)

		act_space = env.action_space
		obs_space = env.observation_space

		_encoder = data['encoder']
		train_encoder = _encoder.get_weight_tied_copy(observation_input=train_obs)
		val_encoder = _encoder.get_weight_tied_copy(observation_input=val_obs)

		# import pdb; pdb.set_trace()

		_regressor = OneLayer(
			BOX_POS_DIM,
			observation_dim=_encoder.output_dim,
		)

		training_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='one_layer')
		sess.run(tf.variables_initializer(training_vars))

		train_regressor = _regressor.get_weight_tied_copy(observation_input=train_encoder.output)
		val_regressor = _regressor.get_weight_tied_copy(observation_input=val_encoder.output)

		train_loss = tf.reduce_mean(tf.square(train_regressor.output - train_joint))
		val_loss = tf.reduce_mean(tf.square(val_regressor.output - val_joint))
		train_summary = tf.summary.scalar('train/loss', train_loss)
		val_summary = tf.summary.scalar('val/loss', val_loss)

		temp_vars = set(tf.all_variables())
		train_opt = tf.train.AdamOptimizer(1e-3).minimize(train_loss, var_list=training_vars)
		optimizer_vars = set(tf.all_variables()) - temp_vars

		# Init variables
		sess.run(tf.variables_initializer(set(training_vars) | optimizer_vars))

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)

		try:
			for timestep in itertools.count():
				if timestep % 500 == 0:
					print ("==== Itr: %d ====" % timestep)
					_, train_loss_val, val_loss_val = sess.run(
						[train_opt, train_loss, val_loss]
					)
					print ("Training loss: ", train_loss_val)
					print ("Validation loss: ", val_loss_val)
					data_dict = dict(
						env=env,
						encoder=_encoder,
						regressor=_regressor,
					)
					joblib.dump(data_dict, args.output_file+'/itr_%d.pkl'%timestep, compress=3)
					# print ("Saved model to %s" % args.output_file)
				else:
					sess.run(train_opt)

		except KeyboardInterrupt:
			print ("Ended training")
			pass

		coord.join(threads)
		sess.close()


if __name__ == '__main__':
	main()