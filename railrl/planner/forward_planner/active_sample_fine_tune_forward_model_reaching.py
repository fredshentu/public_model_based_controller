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


import time
import numpy as np
from rllab.misc import logger
from os import listdir
import os.path as osp
filename = str(uuid.uuid4())
from railrl.data_management.simple_replay_pool import SimpleReplayPool


def save_snapshot(encoder, inverse_model, forward_model, tfmodel_path):
	save_dict = dict(
		encoder=encoder,
		inverse_model=inverse_model,
		forward_model=forward_model
	)

	joblib.dump(save_dict, tfmodel_path, compress=3)
	logger.log("Saved ICM model to {}".format(tfmodel_path))



if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('dynamic_model_path', type=str,
						help='path to the dynamix_model')
	parser.add_argument('env_name', type = str,
						help='environment name')
	parser.add_argument('policy_path', type = str,
						help='path to policy')
	parser.add_argument('save_model_path', type=str,
						help='path to save the fine tuned model')
	parser.add_argument('tf_board', type=str,
						help='path tf_board')
	parser.add_argument('horizon', type=int, help='The horizon of FW model controller')
	parser.add_argument('--variable_horizon', action='store_true', help='if the horizon is variable')
	parser.add_argument('--init_lr', type=float, default=1e-3,
						help='fine tune initial learning_rate')
						
	args = parser.parse_args()
	import gym
	env = gym.make(args.env_name)
	import tensorflow as tf
	from planner import ClippedSgdForwardModelPlanner, InverseModelPlanner, \
					ConstrainedForwardModelPlanner, SgdForwardModelPlanner,\
					FastClippedSgdForwardModelPlanner, FastClippedSgdShootingForwardModelPlanner
	from railrl.predictors.dynamics_model import NoEncoder, FullyConnectedEncoder, ConvEncoder, InverseModel, ForwardModel
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)

	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		##load policy
		policy_data = joblib.load(args.policy_path)
		policy = policy_data['policy']
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
														
		loss = tf.reduce_mean(tf.square(encoder2.output - forward_model.output))
		with tf.variable_scope("new_optimizer"):
			fine_tune_opt = tf.train.AdamOptimizer(args.init_lr).minimize(loss)
		variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope='new_optimizer')
		sess.run(tf.initialize_variables(variables))
		#summary
		summary_writer = tf.summary.FileWriter(args.tf_board, graph = tf.get_default_graph())
		forward_loss_summ = tf.summary.scalar("forward_loss", loss)
		summary = tf.summary.merge_all()
		
		controller = FastClippedSgdShootingForwardModelPlanner(_forward_model, _encoder, env, \
										sess = sess, pos_only = True)
		
		#S_init = env.reset()
		#S_goal = policy run 14 steps
		
		S_goal_list = []
		NUM_GOALS = 1000
		for i in range(NUM_GOALS):
			if i %100 == 0:
				print("sampling policy S_goal")
			obs = env.reset()
			for j in range(50):
				action, _ = policy.get_action(obs)
				obs, r, d, _ = env.step(action)
			S_goal_list.append(obs)
		#collect 5000 goals
		S_goal_list = np.array(S_goal_list)
		for i in range(30000):
			print(i)
			obs = env.reset()
			S_goal = S_goal_list[np.random.randint(NUM_GOALS)]
			#roll out 20 steps using dynamic model controller
			for j in range(30):
				if not args.variable_horizon:
					action, _ = controller.get_action(obs, S_goal, steps = args.horizon)
				else:
					action, _ = controller.get_action(obs, S_goal, steps = max(15-j, 1))
				replay_buffer.add_sample(obs, action, 0, False, False)
				obs, r, d, _ = env.step(action)
				if (i*30 + j) % 5000 == 0:
						PATH = args.save_model_path +"/fine_tune_itr{}.pkl".format(i*20+j)
						save_snapshot(_encoder, _inverse_model, _forward_model, PATH)
				if replay_buffer.size > 500:
					# print("Start Training")
					batch = replay_buffer.random_batch(256)
					obs_batch = list(batch['observations'])
					action_batch = list(batch['actions'])
					next_obs_batch = list(batch['next_observations'])
					# import pdb; pdb.set_trace()
					feed_dict = {s1_ph:obs_batch, s2_ph:next_obs_batch, action_ph:action_batch}
					if (i*30 + j) % 200 == 0:
						_, summ = sess.run([fine_tune_opt, summary], feed_dict = feed_dict)
						summary_writer.add_summary(summ, i*20+j)
					else:
						sess.run(fine_tune_opt, feed_dict = feed_dict)
					

				
				
				# env.render()
			replay_buffer.add_sample(obs, np.zeros(4), 0, True, True)
			
					
			
				