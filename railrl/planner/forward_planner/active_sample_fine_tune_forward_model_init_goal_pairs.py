"""
Fine tune forward model using data generate by itself.

choose S_init, S_goal
	controller gives a_1
	excute a_1, get transition S_init, a_1, S_next
	use this transition to renforce the forward model

Choose S_init, S_goal
	1. using Policy (Fine tune in policy distribution)
	2. randomly choose one (if not state space, need to use replay buffer)
	
	
	python active_sample_fine_tune_forward_model.py ~/Desktop/forward_planner_data/tf_model/Box3d_reach_table_v1_1500k_random_data_1e-3  Box3dReachTable-v1  ~/rllab/data/local/trpo-box3d-state-reach-table-v1-tf-5000itr/trpo_box3d_state_reach_table_v1_tf_5000itr_2017_08_14_22_59_20_0001/itr_3600.pkl ~/Desktop/forward_planner_data/tf_model/Box3d_reach_table_v1_1500k_random_data_1e-3_fine_tune_step1/ ~/Desktop/forward_planner_data/tf_board/Box3d_reach_table-v1_1500k_random_1e-3_fine_tune_step1 1
"""

import argparse
import joblib
import uuid
import pickle

import time
import numpy as np
from rllab.misc import logger
from os import listdir
import os.path as osp
filename = str(uuid.uuid4())
from railrl.data_management.simple_replay_pool import SimpleReplayPool
def env_set_state(env, ob):
	qpos_idx = env.env.init_qpos.shape[0]
	env.env.set_state(ob[:qpos_idx], ob[qpos_idx:])
	return

def save_snapshot(encoder, inverse_model, forward_model, tfmodel_path):
	save_dict = dict(
		encoder=encoder,
		inverse_model=inverse_model,
		forward_model=forward_model
	)

	joblib.dump(save_dict, tfmodel_path, compress=3)
	logger.log("Saved ICM model to {}".format(tfmodel_path))
#rescale from real state to env obs
def rescale_obs(env_name, state):
	if env_name == 'Box3dPush-v2':
		result = state.copy()
		result[4:7] = result[4:7]*10
		return result
	else:
		return None
		

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('dynamic_model_path', type=str,
						help='path to the dynamix_model')
	parser.add_argument('env_name', type = str,
						help='environment name')
	parser.add_argument('init_goal_path', type = str,
						help='path to init_and_goal')
	parser.add_argument('save_model_path', type=str,
						help='path to save the fine tuned model')
	parser.add_argument('tf_board', type=str,
						help='path tf_board')
	parser.add_argument('--horizon', type=int, default = 15, help='The horizon of FW model controller')
	parser.add_argument('--init_lr', type=float, default=1e-3,
						help='fine tune initial learning_rate')
						
	args = parser.parse_args()
	import gym
	env = gym.make(args.env_name)
	env.reset()
	import tensorflow as tf
	from planner import ClippedSgdForwardModelPlanner, InverseModelPlanner, \
					ConstrainedForwardModelPlanner, SgdForwardModelPlanner,\
					FastClippedSgdForwardModelPlanner, FastClippedSgdShootingForwardModelPlanner, CEMPlanner
	from railrl.predictors.dynamics_model import NoEncoder, FullyConnectedEncoder, ConvEncoder, InverseModel, ForwardModel
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

		##load dynamic model
		data = joblib.load(args.dynamic_model_path)
		_encoder = data['encoder']
		_inverse_model = data['inverse_model']
		_forward_model = data['forward_model']

		replay_buffer = SimpleReplayPool(500000, env.observation_space.shape, env.action_space.shape)
		
		action_ph =tf.placeholder(tf.float32, [None,4])
		s1_ph = tf.placeholder(tf.float32, [None] + list(env.observation_space.shape))
		s2_ph = tf.placeholder(tf.float32, [None] + list(env.observation_space.shape))
		encoder1 = _encoder.get_weight_tied_copy(observation_input=s1_ph)
		encoder2 = _encoder.get_weight_tied_copy(observation_input=s2_ph)
		forward_model = _forward_model.get_weight_tied_copy(feature_input=encoder1.output, 
														action_input=action_ph)
		inverse_model = _inverse_model.get_weight_tied_copy(feature_input1=encoder1.output,
														feature_input2=encoder2.output)
														
		loss_dim = tf.reduce_mean(tf.square(encoder2.output - forward_model.output), axis = 0)
		loss = tf.reduce_mean(loss_dim)
		arm_loss = tf.reduce_mean(loss_dim[:4])
		box_loss = tf.reduce_mean(loss_dim[4:6])
		with tf.variable_scope("new_optimizer"):
			fine_tune_opt = tf.train.AdamOptimizer(args.init_lr).minimize(loss)
		variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope='new_optimizer')
		sess.run(tf.initialize_variables(variables))
		#summary
		summary_writer = tf.summary.FileWriter(args.tf_board, graph = tf.get_default_graph())
		forward_loss_summ = tf.summary.scalar("forward_loss", loss)
		forward_arm_loss_summ = tf.summary.scalar("arm_loss", arm_loss)
		forward_box_loss_summ = tf.summary.scalar("box_loss", box_loss)
		
		summary = tf.summary.merge_all()
		
		controller = CEMPlanner(_forward_model, _encoder, env, \
										sess = sess, pos_only = True)
		
		#S_init = env.reset()
		#S_goal = policy run 14 steps
		with open(args.init_goal_path, 'rb') as handle:
			data = pickle.load(handle)
			S_init_state_list = data['S_init']
			S_goal_state_list = data['S_goal']
		for i in range(20000):
			print(i)
			index = np.random.randint(len(S_init_state_list))
			S_init = S_init_state_list[index]
			S_goal = S_goal_state_list[index]
			# print("S_goal")
			# env_set_state(env, S_goal)
			# env.render()
			# time.sleep(0.3)
			env_set_state(env, S_init)
			S_goal = rescale_obs(args.env_name, S_goal)
			S_init = rescale_obs(args.env_name, S_init)
			obs = S_init
	
			#roll out 70 steps using dynamic model controller
			for j in range(70):
				if np.random.rand() < 0.3:
					action = env.action_space.sample()
				else:
					action, _ = controller.get_action(obs, S_goal, steps = args.horizon)
					
				replay_buffer.add_sample(obs, action, 0, False, False)
				obs, r, d, _ = env.step(action)
				# env.render()
				if replay_buffer.size > 500:
					# print("Start Training")
					batch = replay_buffer.random_batch(256)
					obs_batch = list(batch['observations'])
					action_batch = list(batch['actions'])
					next_obs_batch = list(batch['next_observations'])
					# import pdb; pdb.set_trace()
					feed_dict = {s1_ph:obs_batch, s2_ph:next_obs_batch, action_ph:action_batch}
					if (j) == 0:
						_, summ = sess.run([fine_tune_opt, summary], feed_dict = feed_dict)
						summary_writer.add_summary(summ, i)
					else:
						sess.run(fine_tune_opt, feed_dict = feed_dict)
			
			if (i) % 300 == 0:
				PATH = args.save_model_path +"/fine_tune_itr{}.pkl".format(i)
				save_snapshot(_encoder, _inverse_model, _forward_model, PATH)
				
				
				# env.render()
			replay_buffer.add_sample(obs, np.zeros(4), 0, True, True)
			
					
			
				