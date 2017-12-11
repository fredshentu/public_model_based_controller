from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc import logger

from railrl.predictors.dynamics_model import NoEncoder, FullyConnectedEncoder, ConvEncoder, InverseModel, ForwardModel

import argparse
import tensorflow as tf

from os import listdir
import os.path as osp
import joblib

"""
This inverse model takes 2 real gray-scale image, 2 masks of gripper corresponding to those two
real images and 2 blobs which indicates the goal gripper pos

Output: 4 delta_joint angle

architecture of this model:
128*128*6 input -> conv layer -> embedding -> conditional inverse model
"""

########
#Some hyper parameters:
MAX_LENGTH = 10
ACTION_DIM = 4
IM_SIZE = [128,128]
ACTION_SCALE = 10 #scale action to avoid the numerical issue, not sure if putting a tanh is necessary here
########

def read_and_decode(filename_queue, obs_shape):
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(
		  serialized_example,
		  # images: 11*128*128*6
		  # action: 10*4
		  features={
		          'images_list': tf.FixedLenFeature([], tf.string)
		          'action_list': tf.FixedLenFeature([ACTION_DIM], tf.float32)
		  })
	
	

	# Convert label from a scalar uint8 tensor to an int32 scalarself.
	images_list = tf.decode_raw(feature['images_list'], tf.uint8)
	images_list = tf.reshape(images_list, [MAX_LENGTH + 1] + IM_SIZE + [6])
	action_list = tf.cast(features['action_list'], tf.float32)
	action_list = tf.reshape(actino_list, [MAX_LENGTH+1, ACTION_DIM])
	
	#Fist one is always the init_state, choose a goal randomly
	index = np.random.randint(1m)
	#rescale action
	
	
	return 


def inputs(filenames, obs_shape, train=True, batch_size=256, num_epochs = None):
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
				
		obs, next_obs, action = read_and_decode(filename_queue, obs_shape)

		if train:
			num_thread = 10
			queue_capacity = 100000
		else:
			num_thread = 2
			queue_capacity = 50000
		obs, next_obs, actions = tf.train.batch([obs, next_obs, action],\
										batch_size = batch_size, num_threads = num_thread,\
										capacity = queue_capacity, enqueue_many =False)
		#if need to pre-process, add some tf code here
		
		return obs, next_obs, actions