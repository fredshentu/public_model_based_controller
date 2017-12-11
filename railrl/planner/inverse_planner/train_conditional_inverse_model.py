from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc import logger

from railrl.predictors.dynamics_model import NoEncoder, FullyConnectedEncoder, ConvEncoder, \
		InverseModel, ForwardModel, ConditionalInverseModel, ConditionalFeedForwardNet

import argparse
import tensorflow as tf

from os import listdir
import os.path as osp
import joblib


########
#Some hyper parameters:
MAX_LENGTH = 10
ACTION_DIM = 4
IM_SHAPE = [128,128,2]
ACTION_SCALE = 10 #scale action to avoid the numerical issue, not sure if putting a tanh is necessary here
INPUT_CHANNEL = 6
FEATURE_DIM = 128
########

def save_snapshot(encoder_im, encoder_gripper_pos, inverse_model, tfmodel_path):
    save_dict = dict(
        encoder_im=encoder_im,
        inverse_model=inverse_model,
        encoder_gripper_pos=encoder_gripper_pos,
    )

    joblib.dump(save_dict, tfmodel_path, compress=3)

def read_and_decode(filename_queue, obs_shape, step=10):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
          serialized_example,
          # Defaults are not specified since both keys are required.
          features={
                'obs': tf.FixedLenFeature([], tf.string),
                'state': tf.FixedLenFeature([4*step], tf.string)
                'action': tf.FixedLenFeature([4*step], tf.float32),
          })
    
    obs = tf.decode_raw(features['obs'], tf.uint8)
    next_obs = tf.decode_raw(features['next_obs'], tf.uint8)


    
    obs = tf.reshape(obs, [step+1] + obs_shape)
    obs = tf.cast(obs, tf.float32) * (1. / 255) - 0.5


    # Convert label from a scalar uint8 tensor to an int32 scalar.
    state = tf.cast(features['state'], tf.float32)
    action = tf.cast(features['action'], tf.float32)

    state = tf.reshape(state, [step + 1, 4]) # first state as well
    action = tf.reshape(action, [step, 4])*ACTION_SCALE #we always only use the first action


    return obs[0], obs[np.random.randint(1,step + 1)], action, state

	


def inputs(filenames, shape, step, train=True, batch_size=128, num_epochs = None):
	with tf.name_scope('input'):
		filename_queue = tf.train.string_input_producer(
				filenames, num_epochs=num_epochs)
				
		raw_im_init, raw_im_goal, action, state = read_and_decode(filename_queue, shape, step)

		if train:
			num_thread = 10
			queue_capacity = 100000
		else:
			num_thread = 2
			queue_capacity = 50000
		raw_im_init_raw, raw_im_goal_batch, action_batch, state_batch = \
										tf.train.batch([inverse_model_input, action],\
										batch_size = batch_size, num_threads = num_thread,\
										capacity = queue_capacity, enqueue_many =False)

		return raw_im_init_batch, raw_im_goal_batch, action_batch, state_batch

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument('dataset_path', type=str, 
						help="path of training and validation dataset")
	parser.add_argument('segmentor_path', type=str,
						help = "path to the segmentation net work")
	parser.add_argument('tfboard_path', type=str, default='/tmp/tfboard')
	parser.add_argument('tfmodel_path', type=str, default='/tmp/tfmodels')
	parser.add_argument('--init_lr', type=float, default=1e-3)
	parser.add_argument('--num_itr', type=int, default=20000)
	parser.add_argument('--log_freq', type=int, default=10)
	parser.add_argument('--save_freq', type=int, default=1000)
	
	args = parser.parse_args()
	dataset_names = list(map(lambda file_name: osp.join(args.dataset_path, file_name), listdir(args.dataset_path)))
	#No validation set since validating model can not tell anything about the model
	raw_im_init_batch, raw_im_goal_batch, action_batch, state_batch = \
										inputs(dataset_names, IM_SHAPE, MAX_LENGTH, train = True)
	###########################
	#segment network preprocess data...
	#get x1,y1 and x2,y2 for goal and x1,y1,x2,y2 for init
	data = joblib.load(args.segmentor_path)
	_segmentor = data['segmentor']
	_segmentor_encoder = data['encoder']
	init_im_segmentor_encoder = _segmentor_encoder.get_weight_tied_copy(observation_input=raw_im_init_batch)
	goal_im_segmentor_encoder = _segmentor_encoder.get_weight_tied_copy(observation_input=raw_im_goal_batch)
	init_im_segmentor = _segmentor.get_weight_tied_copy(feature_input = init_im_segmentor_encoder.unflattend_output)
	goal_im_segmentor = _segmentor.get_weight_tied_copy(feature_input = goal_im_segmentor_encoder.unflattend_output)
	
	
	init_cam_centroid = tf.concat([init_im_segmentor.cam1_centroid, init_im_segmentor.cam2_centroid], axis = 1)
	goal_cam_centroid = tf.concat([goal_im_segmentor.cam1_centroid, goal_im_segmentor.cam2_centroid], axis = 1)
	goal_valid = goal_im_segmentor.valid
	

	blobs_and_state = tf.concat([init_cam_centroid, goal_cam_centroid, state_batch])
	###########################
	_encoder_im = ConvEncoder(
		feature_dim = FEATURE_DIM,
		input_shape = IM_SHAPE,
		conv_filters=(64, 64, 32, 32, 16),
		conv_filter_sizes=((3,3), (3,3), (3,3), (3,3), (3,3)),
		conv_strides=(2, 2, 2, 2, 2),
		conv_pads=('SAME', 'SAME', 'SAME', 'SAME', 'SAME'),
		hidden_sizes=(),
		hidden_activation=tf.nn.elu,
		)
	_encoder_gripper_pos = FullyConnectedEncoder(
		feature_dim = FEATURE_DIM,
		hidden_sizes = (),
		observation_dim = 8+4, #4 xy pairs + 4 joint current state
		)
		
	_inverse_model = ConditionalInverseModel(
		feature_dim = FEATURE_DIM,
		hidden_sizes=(200,200),
		action_dim = ACTION_DIM,
		hidden_activation = tf.nn.relu,
		output_activation = None,
		)
	
	sess = tf.Session()
	_encoder_im.sess = sess
	_encoder_gripper_pos.sess = sess
	_inverse_model.sess = sess
	
	with sess.as_default():
		sess.run(tf.initialize_all_variables())
		encoder_im = _encoder_im.get_weight_tied_copy(observation_input=raw_im_init_batch)
		encoder_gripper_pos = _encoder_gripper_pos.get_weight_tied_copy(observation_input=blobs_and_state)
		inverse_model =  _inverse_model.get_weight_tied_copy(feature_input1 = encoder_im.output, \
												feature_input2 = encoder_gripper_pos.output)
		raw_loss = tf.square(inverse_model.output - train_action)
		valid_loss =  raw_loss * goal_valid
		loss_dim = tf.reduce_mean(valid_loss, axis = 0)
		total_loss = tf.reduce_mean(loss_dim)
		train_opt= tf.train.AdamOptimizer(args.init_lr).minimize(total_loss)
		
		summary_writer = tf.summary.FileWriter(args.tfboard_path, graph=tf.get_default_graph())
		train_loss_summ = []
		for i in range(ACTION_DIM):
			train_loss_summ.append(tf.summary.scalar("train/action_dim_{} loss".format(i+1), loss_dim[i]))
		train_summary_op = tf.summary.merge(train_loss_summ)
		sess.run(tf.initialize_all_variables())
		
		coord = tf.train.Coordinator()
		try:
			for timestep in range(args.num_itr):
				if timestep % args.log_freq == 0:
					logger.log("Start itr {}".format(timestep))
					_, train_summary = sess.run(
						[train_opt, train_summary_op]
					)
					summary_writer.add_summary(train_summary, timestep)
				else:
					sess.run(train_opt)
					
				if timestep % args.save_freq == 0:
					save_snapshot(_encoder_im, _incoder_gripper_pos, _inverse_model, args.tfmodel_path)
		except KeyboardInterrupt:
			print ("End training...")
			pass

		coord.join(threads)
		sess.close()
		
		
