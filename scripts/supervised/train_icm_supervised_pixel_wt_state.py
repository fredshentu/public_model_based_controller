"""
Author: Dian Chen
Note that this script only applies to Box3dReachPixel environments
"""

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


ACTION_DIM_MAP = {
	"Box3dReachPixel-v11" : 4,
	"real_robot": 4,
}

OBS_SHAPE_MAP = {
	"Box3dReachPixel-v11" : [84, 84, 8],
	"real_robot": [128, 128, 6],
}

def read_and_decode(filename_queue, obs_shape):
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(
		  serialized_example,
		  # Defaults are not specified since both keys are required.
		  features={
				  'obs': tf.FixedLenFeature([], tf.string),
				  'next_obs': tf.FixedLenFeature([], tf.string),
				  'action': tf.FixedLenFeature([4], tf.float32),
				  'state': tf.FixedLenFeature([8], tf.float32),
				  'next_state': tf.FixedLenFeature([8], tf.float32),
		  })
	
	obs = tf.decode_raw(features['obs'], tf.uint8)
	next_obs = tf.decode_raw(features['next_obs'], tf.uint8)
	
	obs = tf.reshape(obs, obs_shape)
	next_obs = tf.reshape(next_obs, obs_shape)

	obs = tf.cast(obs, tf.float32) * (1. / 255) - 0.5
	next_obs = tf.cast(next_obs, tf.float32) * (1. / 255) - 0.5

	# Convert label from a scalar uint8 tensor to an int32 scalar.
	action = tf.cast(features['action'], tf.float32) * [1./1023, 1./249, 1./249, 1./1023]
	state = tf.cast(features['state'], tf.float32) / 2048.0
	next_state = tf.cast(features['next_state'], tf.float32) / 2048.0
	
	return obs, next_obs, action, state, next_state


def inputs(filenames, obs_shape, train=True, batch_size=128, num_epochs = None):

	with tf.name_scope('input'):
		filename_queue = tf.train.string_input_producer(
				filenames, num_epochs=num_epochs)
				
		image, next_image, action, state, next_state = read_and_decode(filename_queue, obs_shape)

		if train:
			num_thread = 12
			queue_capacity = 40000
		else:
			num_thread = 4
			queue_capacity = 4000
		images, next_images, actions, states, next_states = tf.train.batch([image, next_image, action, state, next_state],\
										batch_size = batch_size, num_threads = num_thread,\
										capacity = queue_capacity, enqueue_many =False)

		
		return images, next_images, actions, states, next_states



def save_snapshot(encoder, inverse_model, forward_model, state_encoder, tfmodel_path):
	save_dict = dict(
		encoder=encoder,
		inverse_model=inverse_model,
		forward_model=forward_model,
		state_encoder=state_encoder,
	)

	joblib.dump(save_dict, tfmodel_path, compress=3)
	logger.log("Saved ICM model to {}".format(tfmodel_path))


def cos_loss(A, B):
	dotproduct = tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(A, 1), tf.nn.l2_normalize(B,1)), axis = 1)
	return 1 - tf.reduce_mean(dotproduct)


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('env_name', type=str, 
						help="name of gym env")
	parser.add_argument('dataset_path', type=str, 
						help="path of training and validation dataset")
	parser.add_argument('--tfboard_path', type=str, default='/tmp/tfboard')
	parser.add_argument('--tfmodel_path', type=str, default='/tmp/tfmodels')
	parser.add_argument('--restore', action='store_true')
	# Training parameters
	parser.add_argument('--val_ratio', type=float, default=0.05,
						help="ratio of validation sets")
	parser.add_argument('--num_itr', type=int, default=10000000)
	parser.add_argument('--val_freq', type=int, default=200)
	parser.add_argument('--log_freq', type=int, default=50)
	parser.add_argument('--save_freq', type=int, default=5000)

	# ICM parameters
	parser.add_argument('--init_lr', type=float, default=1e-4)
	parser.add_argument('--forward_weight', type=float, default=0.8,
						help="the ratio of forward loss vs inverse loss")
	parser.add_argument('--cos_forward', action='store_true',
						help="whether to use cosine forward loss")

	args = parser.parse_args()


	# Get dataset
	dataset_names = list(map(lambda file_name: osp.join(args.dataset_path, file_name), listdir(args.dataset_path)))
	val_set_names = dataset_names[:int(len(dataset_names)*args.val_ratio)]
	train_set_names = dataset_names[int(len(dataset_names)*args.val_ratio):]

	train_queue = tf.train.string_input_producer(train_set_names, num_epochs=None)
	val_queue = tf.train.string_input_producer(val_set_names, num_epochs=None)

	obs_shape = OBS_SHAPE_MAP[args.env_name]
	action_dim = ACTION_DIM_MAP[args.env_name]
	train_obs, train_next_obs, train_action, train_state, train_next_state = inputs(train_set_names, obs_shape, train=True)
	val_obs, val_next_obs, val_action, val_state, val_next_state = inputs(val_set_names, obs_shape, train=False)

	#not yet implemented
	if args.restore:
		raise NotImplementedError
		models_dict = joblib.load(args.tfmodel_path)
		_encoder = models_dict['encoder']
		_inverse_model = model.dict['inverse_model']
		_forward_model = model.dict['forward_model']
	else:
		_encoder = ConvEncoder(
			feature_dim=256,
			input_shape=obs_shape,
			conv_filters=(64, 64, 64, 64, 32),
			conv_filter_sizes=((5,5), (5,5), (5,5), (5,5), (3,3)),
			conv_strides=(2, 2, 2, 2, 2),
			conv_pads=('SAME', 'SAME', 'SAME', 'SAME', 'SAME'),
			hidden_sizes=(256,),
			hidden_activation=tf.nn.elu,
		)
		_state_encoder = FullyConnectedEncoder(
			200,
			observation_dim = 8,
			hidden_sizes = (256,),
			hidden_activation = tf.nn.elu,
		)
		#TODO: add one more encoder before inverse model?
		_inverse_model = InverseModel(
			feature_dim=256+200,
			action_dim=action_dim,
			hidden_sizes=(256,),
			hidden_activation=tf.nn.elu,
			output_activation=tf.nn.tanh,
		)
		_forward_model = ForwardModel(
			feature_dim=256 + 200,
			action_dim=action_dim,
			hidden_sizes=(256,),
			hidden_activation=tf.nn.elu,
		)

	sess = tf.Session()
	_encoder.sess = sess
	_inverse_model.sess = sess
	_forward_model.sess = sess
	_state_encoder.sess = sess
	
	with sess.as_default():
		# Initialize variables for get_copy to work
		sess.run(tf.initialize_all_variables())

		train_encoder1 = _encoder.get_weight_tied_copy(observation_input=train_obs)
		train_encoder2 = _encoder.get_weight_tied_copy(observation_input=train_next_obs)
		train_state_encoder1 = _state_encoder.get_weight_tied_copy(observation_input=train_state)
		train_state_encoder2 = _state_encoder.get_weight_tied_copy(observation_input=train_next_state)
		train_feature1 = tf.concat(1, [train_encoder1.output, train_state_encoder1.output])
		train_feature2 = tf.concat(1, [train_encoder2.output, train_state_encoder2.output])
		train_inverse_model = _inverse_model.get_weight_tied_copy(feature_input1=train_feature1, feature_input2=train_feature2)
		train_forward_model = _forward_model.get_weight_tied_copy(feature_input=train_feature1, action_input=train_action)

		
		val_encoder1 = _encoder.get_weight_tied_copy(observation_input=val_obs)
		val_encoder2 = _encoder.get_weight_tied_copy(observation_input=val_next_obs)
		val_state_encoder1 = _state_encoder.get_weight_tied_copy(observation_input=val_state)
		val_state_encoder2 = _state_encoder.get_weight_tied_copy(observation_input=val_next_state)
		val_feature1 = tf.concat(1, [val_encoder1.output, val_state_encoder1.output])
		val_feature2 = tf.concat(1, [val_encoder2.output, val_state_encoder2.output])
		val_inverse_model = _inverse_model.get_weight_tied_copy(feature_input1=val_feature1, feature_input2=val_feature2)
		val_forward_model = _forward_model.get_weight_tied_copy(feature_input=val_feature1, action_input=val_action)
		
		if args.cos_forward:
			train_forward_loss = cos_loss(train_feature2, train_forward_model.output)
			val_forward_loss = cos_loss(val_feature2, val_forward_model.output)
		else:
			train_forward_loss = tf.reduce_mean(tf.square(train_feature2 - train_forward_model.output))
			val_forward_loss = tf.reduce_mean(tf.square(val_feature2 - val_forward_model.output))

		train_inverse_losses = tf.reduce_mean(tf.square(train_action - train_inverse_model.output), axis=0)
		val_inverse_losses = tf.reduce_mean(tf.square(val_action - val_inverse_model.output), axis=0)
		train_inverse_separate_summ = []
		val_inverse_separate_summ = []
		for joint_idx in range(action_dim):
			train_inverse_separate_summ.append(tf.summary.scalar("train/icm_inverse_loss/joint_{}".format(joint_idx), train_inverse_losses[joint_idx]))
			val_inverse_separate_summ.append(tf.summary.scalar("val/icm_inverse_loss/joint_{}".format(joint_idx), val_inverse_losses[joint_idx]))
		
		train_inverse_loss = tf.reduce_mean(train_inverse_losses)
		val_inverse_loss = tf.reduce_mean(val_inverse_losses)
		
		train_total_loss = args.forward_weight * train_forward_loss + (1. - args.forward_weight) * train_inverse_loss
		val_total_loss = args.forward_weight * val_forward_loss + (1. - args.forward_weight) * val_inverse_loss
		icm_opt = tf.train.AdamOptimizer(args.init_lr).minimize(train_total_loss)

		# Setup summaries
		summary_writer = tf.summary.FileWriter(args.tfboard_path, graph=tf.get_default_graph())

		train_inverse_loss_summ = tf.summary.scalar("train/icm_inverse_loss/total_mean", train_inverse_loss)
		train_forward_loss_summ = tf.summary.scalar("train/icm_forward_loss", train_forward_loss)
		train_total_loss_summ = tf.summary.scalar("train/icm_total_loss", train_total_loss)
		val_inverse_loss_summ = tf.summary.scalar("val/icm_inverse_loss/total_mean", val_inverse_loss)
		val_forward_loss_summ = tf.summary.scalar("val/icm_forward_loss", val_forward_loss)
		val_total_loss_summ = tf.summary.scalar("val/icm_total_loss", val_total_loss)

		train_summary_op = tf.summary.merge(
			[train_inverse_loss_summ, 
			 train_forward_loss_summ,
			 train_total_loss_summ] + train_inverse_separate_summ)
		val_summary_op = tf.summary.merge(
			[val_inverse_loss_summ, 
			 val_forward_loss_summ,
			 val_total_loss_summ] + val_inverse_separate_summ)

		logger.log("Finished creating ICM model")

		sess.run(tf.initialize_all_variables())

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		
		try:
			for timestep in range(args.num_itr):
				if timestep % args.log_freq == 0:
					logger.log("Start itr {}".format(timestep))
					_, train_summary = sess.run(
						[icm_opt, train_summary_op]
					)
					summary_writer.add_summary(train_summary, timestep)
				else:
					sess.run(icm_opt)

				if timestep % args.save_freq == 0:
					save_snapshot(_encoder, _inverse_model, _forward_model, _state_encoder, args.tfmodel_path)
				
				if timestep % args.val_freq == 0:
					val_summary = sess.run(
						val_summary_op
					)
					summary_writer.add_summary(val_summary, timestep)

		except KeyboardInterrupt:
			print ("End training...")
			pass

		coord.join(threads)
		sess.close()


if __name__ == "__main__":
	main()
	# test_img()


