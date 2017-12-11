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
	"Box3dPush-v0": 4,
	"Box3dPush-v1": 4,
	"Box3dPush-v2": 4,
	"Box3dReachTable-v0": 4,
	"Box3dReachTable-v1": 4,
	"Box3dPush-v2_arm_coord":4,
}

OBS_SHAPE_MAP = {
	"Box3dPush-v0":21,
	"Box3dPush-v1":21,
	"Box3dPush-v2":21,
	"Box3dReachTable-v0": 21,
	"Box3dReachTable-v1": 21,
	"Box3dPush-v2_arm_coord":24,
}

def read_and_decode(filename_queue, obs_shape):
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(
		  serialized_example,
		  # Defaults are not specified since both keys are required.
		  features={
				  'relate_obs': tf.FixedLenFeature([obs_shape], tf.float32),
				  'next_relate_obs': tf.FixedLenFeature([obs_shape], tf.float32),
				  'action': tf.FixedLenFeature([4], tf.float32),
		  })
	
	

	# Convert label from a scalar uint8 tensor to an int32 scalar.
	obs = tf.cast(features['relate_obs'], tf.float32)
	next_obs = tf.cast(features['next_relate_obs'], tf.float32)
	action = tf.cast(features['action'], tf.float32)
	
	return obs, next_obs, action


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
			queue_capacity = 1000000
		else:
			num_thread = 2
			queue_capacity = 50000
		obs, next_obs, actions = tf.train.batch([obs, next_obs, action],\
										batch_size = batch_size, num_threads = num_thread,\
										capacity = queue_capacity, enqueue_many =False)

		
		return obs, next_obs, actions



def save_snapshot(encoder, inverse_model, forward_model, tfmodel_path):
	save_dict = dict(
		encoder=encoder,
		inverse_model=inverse_model,
		forward_model=forward_model
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
	parser.add_argument('val_random_data', type=str, 
						help="path of training and validation dataset")
	parser.add_argument('val_contact_data', type=str, 
						help="path of training and validation dataset")
	parser.add_argument('tfboard_path', type=str, default='/tmp/tfboard')
	parser.add_argument('tfmodel_path', type=str, default='/tmp/tfmodels')
	parser.add_argument('--restore', action='store_true')
	# Training parameters
	parser.add_argument('--num_itr', type=int, default=10000000)
	parser.add_argument('--val_freq', type=int, default=200)
	parser.add_argument('--log_freq', type=int, default=50)
	parser.add_argument('--save_freq', type=int, default=5000)

	# ICM parameters
	parser.add_argument('--init_lr', type=float, default=2e-3)
	parser.add_argument('--forward_weight', type=float, default=0.5,
						help="the ratio of forward loss vs inverse loss")
	parser.add_argument('--cos_forward', action='store_true',
						help="whether to use cosine forward loss")

	args = parser.parse_args()


	# Get dataset
	train_set_names = list(map(lambda file_name: osp.join(args.dataset_path, file_name), listdir(args.dataset_path)))
	val_random_set_names = list(map(lambda file_name: osp.join(args.val_random_data, file_name), listdir(args.val_random_data)))
	val_contact_set_names = list(map(lambda file_name: osp.join(args.val_contact_data, file_name), listdir(args.val_contact_data)))
	# import pdb; pdb.set_trace()
	
	obs_shape = OBS_SHAPE_MAP[args.env_name]
	action_dim = ACTION_DIM_MAP[args.env_name]
	train_obs, train_next_obs, train_action = inputs(train_set_names, obs_shape, train=True)
	val_random_obs, val_random_next_obs, val_random_action = inputs(val_random_set_names, obs_shape, train=False)
	val_contact_obs, val_contact_next_obs, val_contact_action = inputs(val_contact_set_names, obs_shape, train=False)
	
	if args.restore:
		models_dict = joblib.load(args.tfmodel_path)
		_encoder = models_dict['encoder']
		_inverse_model = model.dict['inverse_model']
		_forward_model = model.dict['forward_model']
	else:
		_encoder = NoEncoder(obs_shape, observation_dim = [obs_shape])
		_inverse_model = InverseModel(
			feature_dim=obs_shape,
			action_dim=action_dim,
			hidden_sizes=(256,256),
			hidden_activation=tf.nn.elu,
			output_activation=tf.nn.tanh,
		)
		_forward_model = ForwardModel(
			feature_dim=obs_shape,
			action_dim=action_dim,
			hidden_sizes=(256,257),
			hidden_activation=tf.nn.elu,
		)
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)

	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	_encoder.sess = sess
	_inverse_model.sess = sess
	_forward_model.sess = sess

	with sess.as_default():
		# Initialize variables for get_copy to work
		sess.run(tf.initialize_all_variables())

		train_encoder1 = _encoder.get_weight_tied_copy(observation_input=train_obs)
		train_encoder2 = _encoder.get_weight_tied_copy(observation_input=train_next_obs)
		# import pdb; pdb.set_trace()
		train_inverse_model = _inverse_model.get_weight_tied_copy(feature_input1=train_encoder1.output, feature_input2=train_encoder2.output)
		train_forward_model = _forward_model.get_weight_tied_copy(feature_input=train_encoder1.output, action_input=train_action)

		val_random_encoder1 = _encoder.get_weight_tied_copy(observation_input=val_random_obs)
		val_random_encoder2 = _encoder.get_weight_tied_copy(observation_input=val_random_next_obs)
		val_random_inverse_model = _inverse_model.get_weight_tied_copy(feature_input1=val_random_encoder1.output, feature_input2=val_random_encoder2.output)
		val_random_forward_model = _forward_model.get_weight_tied_copy(feature_input=val_random_encoder1.output, action_input=val_random_action)
		
		val_contact_encoder1 = _encoder.get_weight_tied_copy(observation_input=val_contact_obs)
		val_contact_encoder2 = _encoder.get_weight_tied_copy(observation_input=val_contact_next_obs)
		val_contact_inverse_model = _inverse_model.get_weight_tied_copy(feature_input1=val_contact_encoder1.output, feature_input2=val_contact_encoder2.output)
		val_contact_forward_model = _forward_model.get_weight_tied_copy(feature_input=val_contact_encoder1.output, action_input=val_contact_action)
		
		if args.cos_forward:
			train_forward_loss = cos_loss(train_encoder2.output, train_forward_model.output)
			val_forward_loss = cos_loss(val_encoder2.output, val_forward_model.output)
		else:
			train_forward_loss = tf.reduce_mean(tf.square(train_encoder2.output - train_forward_model.output))
			#use only if in state space!!!!!!!
			train_forward_loss_arm = tf.reduce_mean(tf.square(train_encoder2.output[:,:4] - train_forward_model.output[:,:4]))
			train_forward_loss_box = tf.reduce_mean(tf.square(train_encoder2.output[:,4:7] - train_forward_model.output[:,4:7]))

			val_random_forward_loss = tf.reduce_mean(tf.square(val_random_encoder2.output - val_random_forward_model.output))
			val_random_forward_loss_arm = tf.reduce_mean(tf.square(val_random_encoder2.output[:,:4] - val_random_forward_model.output[:,:4]))
			val_random_forward_loss_box = tf.reduce_mean(tf.square(val_random_encoder2.output[:,4:7] - val_random_forward_model.output[:,4:7]))

			val_contact_forward_loss = tf.reduce_mean(tf.square(val_contact_encoder2.output - val_contact_forward_model.output))
			val_contact_forward_loss_arm = tf.reduce_mean(tf.square(val_contact_encoder2.output[:,:4] - val_contact_forward_model.output[:,:4]))
			val_contact_forward_loss_box = tf.reduce_mean(tf.square(val_contact_encoder2.output[:,4:7] - val_contact_forward_model.output[:,4:7]))

		train_inverse_losses = tf.reduce_mean(tf.square(train_action - train_inverse_model.output), axis=0)
		val_random_inverse_losses = tf.reduce_mean(tf.square(val_random_action - val_random_inverse_model.output), axis=0)
		val_contact_inverse_losses = tf.reduce_mean(tf.square(val_contact_action - val_contact_inverse_model.output), axis=0)
		
		train_inverse_separate_summ = []
		val_random_inverse_separate_summ = []
		val_contact_inverse_separate_summ = []
		for joint_idx in range(action_dim):
			train_inverse_separate_summ.append(tf.summary.scalar("train/icm_inverse_loss/joint_{}".format(joint_idx), train_inverse_losses[joint_idx]))
			val_random_inverse_separate_summ.append(tf.summary.scalar("random_val/icm_inverse_random_loss/joint_{}".format(joint_idx), val_random_inverse_losses[joint_idx]))
			val_contact_inverse_separate_summ.append(tf.summary.scalar("contact_val/icm_inverse_random_loss/joint_{}".format(joint_idx), val_contact_inverse_losses[joint_idx]))
			
		train_inverse_loss = tf.reduce_mean(train_inverse_losses)
		val_random_inverse_loss = tf.reduce_mean(val_random_inverse_losses)
		val_contact_inverse_loss = tf.reduce_mean(val_contact_inverse_losses)
		
		
		train_total_loss = args.forward_weight * train_forward_loss + (1. - args.forward_weight) * train_inverse_loss
		val_random_total_loss = args.forward_weight * val_random_forward_loss + (1. - args.forward_weight) * val_random_inverse_loss
		val_contact_total_loss = args.forward_weight * val_contact_forward_loss + (1. - args.forward_weight) * val_contact_inverse_loss
		
		icm_opt = tf.train.AdamOptimizer(args.init_lr).minimize(train_total_loss)
		_,train_data_forward_var = tf.nn.moments(train_obs, axes = [1])
		_,train_data_box_var = tf.nn.moments(train_obs[:,4:7], axes=[1])
		
		
		# Setup summaries
		summary_writer = tf.summary.FileWriter(args.tfboard_path, graph=tf.get_default_graph())
		train_forward_loss_arm_summ = tf.summary.scalar("train/forward_loss_arm", train_forward_loss_arm)
		train_forward_loss_box_summ = tf.summary.scalar("train/forward_loss_box", train_forward_loss_box)
		train_inverse_loss_summ = tf.summary.scalar("train/icm_inverse_loss/total_mean", train_inverse_loss)
		train_forward_loss_summ = tf.summary.scalar("train/icm_forward_loss", train_forward_loss)
		train_total_loss_summ = tf.summary.scalar("train/icm_total_loss", train_total_loss)
		
		random_val_forward_loss_arm_summ = tf.summary.scalar("random_val/forward_loss_arm", val_random_forward_loss_arm)
		random_val_forward_loss_box_summ = tf.summary.scalar("random_val/forward_loss_box", val_random_forward_loss_box)
		random_val_inverse_loss_summ = tf.summary.scalar("random_val/icm_inverse_loss/total_mean", val_random_inverse_loss)
		random_val_forward_loss_summ = tf.summary.scalar("random_val/icm_forward_loss", val_random_forward_loss)
		random_val_total_loss_summ = tf.summary.scalar("random_val/icm_total_loss", val_random_total_loss)
		
		contact_val_forward_loss_arm_summ = tf.summary.scalar("contact_val/forward_loss_arm", val_contact_forward_loss_arm)
		contact_val_forward_loss_box_summ = tf.summary.scalar("contact_val/forward_loss_box", val_contact_forward_loss_box)
		contact_val_inverse_loss_summ = tf.summary.scalar("contact_val/icm_inverse_loss/total_mean", val_contact_inverse_loss)
		contact_val_forward_loss_summ = tf.summary.scalar("contact_val/icm_forward_loss", val_contact_forward_loss)
		contact_val_total_loss_summ = tf.summary.scalar("contact_val/icm_total_loss", val_contact_total_loss)
		
		forward_data_variance_summ = tf.summary.scalar("training_data_forward_variance", \
															tf.reduce_mean(train_data_forward_var))
		forward_data_box_variance_summ = tf.summary.scalar("training_data_forward_box_variance", \
															tf.reduce_mean(train_data_box_var))
	


		train_summary_op = tf.summary.merge(
			[train_inverse_loss_summ, 
			 train_forward_loss_summ,
			 train_forward_loss_arm_summ,
			 train_forward_loss_box_summ,
			 train_total_loss_summ,
			 forward_data_variance_summ,
			 forward_data_box_variance_summ,] + train_inverse_separate_summ)
			 
		val_summary_op = tf.summary.merge(
			[random_val_forward_loss_arm_summ, 
			 random_val_forward_loss_box_summ,
			 random_val_inverse_loss_summ,
			 random_val_forward_loss_summ,
			 random_val_total_loss_summ,
			 
			 contact_val_forward_loss_arm_summ,
			 contact_val_forward_loss_box_summ,
			 contact_val_inverse_loss_summ,
			 contact_val_forward_loss_summ,
			 contact_val_total_loss_summ,
			 ] + val_random_inverse_separate_summ + val_contact_inverse_separate_summ)

		logger.log("Finished creating ICM model")

		sess.run(tf.initialize_all_variables())

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)

		try:
			for timestep in range(args.num_itr):
				# print(timestep)
				# print(sess.run(train_action))
				# print("wow")
				if timestep % args.log_freq == 0:
					logger.log("Start itr {}".format(timestep))
					_, train_summary = sess.run(
						[icm_opt, train_summary_op]
					)
					summary_writer.add_summary(train_summary, timestep)
				else:
					sess.run(icm_opt)

				if timestep % args.save_freq == 0:
					save_snapshot(_encoder, _inverse_model, _forward_model, args.tfmodel_path)
				
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


