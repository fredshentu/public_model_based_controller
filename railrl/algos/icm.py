"""
:author: Dian Chen
"""

import time

import numpy as np
import tensorflow as tf

from rllab.spaces import Box, Discrete
from rllab.algos.base import RLAlgorithm
from rllab.misc.overrides import overrides
from rllab.misc import logger
from railrl.algos.online_algorithm import OnlineAlgorithm
from railrl.predictors.dynamics_model import NoEncoder, FullyConnectedEncoder, InverseModel, ForwardModel
from railrl.misc.icm_util import save_data

TENSORBOARD_PERIOD = 1000

class ICM(RLAlgorithm):
	"""
	RL with inverse curiosity module
	"""
	def __init__(
			self,
			env,
			algo: OnlineAlgorithm,
			no_encoder=False,
			feature_dim=10,
			forward_weight=0.8,
			external_reward_weight=0.01,
			inverse_tanh=False,
			init_learning_rate=1e-4,
			algo_update_freq=1,
			**kwargs
	):
		"""
		:param env: Environment
		:param algo: Algorithm that will be used with ICM
		:param encoder: State encoder that maps s to f
		:param inverse_model: Inverse dynamics model that maps (f1, f2) to actions
		:param forward_model: Forward dynamics model that maps (f1, a) to f2
		:param forward_weight: Weight from 0 to 1 that balances forward loss and inverse loss
		:param external_reward_weight: Weight that balances external reward and internal reward
		:param init_learning_rate: Initial learning rate of optimizer
		"""
		self.algo = algo
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
		self.sess = self.algo.sess or tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
		self.external_reward_weight = external_reward_weight
		self.summary_writer = self.algo.summary_writer
		self.algo_update_freq = algo_update_freq
		act_space = env.action_space
		obs_space = env.observation_space
		
		# Setup ICM models
		self.s1 = tf.placeholder(tf.float32, [None] + list(obs_space.shape))
		self.s2 = tf.placeholder(tf.float32, [None] + list(obs_space.shape))
		self.asample = tf.placeholder(tf.float32, [None, act_space.flat_dim])
		self.external_rewards = tf.placeholder(tf.float32, (None,))

		if len(obs_space.shape) == 1:
			if no_encoder:
				self._encoder = NoEncoder(obs_space.flat_dim, env_spec=env.spec)
			else:
				self._encoder = FullyConnectedEncoder(feature_dim, env_spec=env.spec)
		else:
			# TODO: implement conv encoder
			raise NotImplementedError("Currently only supports flat observation input!")

		self._encoder.sess = self.sess
		# Initialize variables for get_copy to work
		self.sess.run(tf.initialize_all_variables())
		with self.sess.as_default():
			self.encoder1 = self._encoder.get_weight_tied_copy(observation_input=self.s1)
			self.encoder2 = self._encoder.get_weight_tied_copy(observation_input=self.s2)

		self._inverse_model = InverseModel(feature_dim, env_spec=env.spec)
		self._forward_model = ForwardModel(feature_dim, env_spec=env.spec)
		self._inverse_model.sess = self.sess
		self._forward_model.sess = self.sess
		# Initialize variables for get_copy to work
		self.sess.run(tf.initialize_all_variables())
		with self.sess.as_default():
			self.inverse_model = self._inverse_model.get_weight_tied_copy(feature_input1=self.encoder1.output, 
																		  feature_input2=self.encoder2.output)
			self.forward_model = self._forward_model.get_weight_tied_copy(feature_input=self.encoder1.output,
																	  	  action_input=self.asample)

		# Define losses
		self.forward_loss = tf.reduce_mean(tf.square(self.encoder2.output - self.forward_model.output))
		# self.forward_loss = tf.nn.l2_loss(self.encoder2.output - self.forward_model.output)
		if isinstance(act_space, Box):
			self.inverse_loss = tf.reduce_mean(tf.square(self.asample - self.inverse_model.output))
		elif isinstance(act_space, Discrete):
			# TODO: Implement softmax loss
			raise NotImplementedError
		else:
			raise NotImplementedError
		self.internal_rewards = tf.reduce_sum(tf.square(self.encoder2.output - self.forward_model.output), axis=1)
		self.mean_internal_rewards = tf.reduce_mean(self.internal_rewards)
		self.mean_external_rewards = tf.reduce_mean(self.external_rewards)
		
		self.total_loss = forward_weight * self.forward_loss + \
						(1. - forward_weight) * self.inverse_loss
		self.icm_opt = tf.train.AdamOptimizer(init_learning_rate).\
						minimize(self.total_loss)


		# Setup summaries
		inverse_loss_summ = tf.summary.scalar("icm_inverse_loss", self.inverse_loss)
		forward_loss_summ = tf.summary.scalar("icm_forward_loss", self.forward_loss)
		total_loss_summ = tf.summary.scalar("icm_total_loss", self.total_loss)
		internal_rewards = tf.summary.scalar("mean_internal_rewards", self.mean_internal_rewards)
		external_rewards = tf.summary.scalar("mean_external_rewards_training", self.mean_external_rewards)
		var_summ = []
		for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
			var_summ.append(tf.summary.histogram(var.op.name, var))
		self.summary = tf.summary.merge([inverse_loss_summ, forward_loss_summ, total_loss_summ,\
							internal_rewards, external_rewards])
		# Initialize variables
		
		self.sess.run(tf.initialize_all_variables())


	@overrides
	def train(self):
		with self.sess.as_default():
			self.algo._init_training()
			self.algo._start_worker()
			self.algo._switch_to_training_mode()

			observation = self.algo.training_env.reset()
			self.algo.exploration_strategy.reset()
			itr = 0
			path_length = 0
			path_return = 0
			for epoch in range(self.algo.n_epochs):
				logger.push_prefix('Epoch #%d | ' % epoch)
				logger.log("Training started")
				start_time = time.time()
				for t in range(self.algo.epoch_length):
					with self.algo._eval_then_training_mode():
						# Bug here!!!!!!
						# action, _ = self.algo.policy.get_action(observation)
						action = self.algo.exploration_strategy.get_action(itr,
																		   observation,
																		   self.algo.policy)
					if self.algo.render:
						self.algo.training_env.render()
					next_ob, raw_reward, terminal, _ = self.algo.training_env.step(
						self.algo.process_action(action)
					)
					# Some envs return a Nx1 vector for the observation
					next_ob = next_ob.flatten()
					reward = raw_reward * self.algo.scale_reward
					
					# # JUST FOR DEBUG: save data
					# save_data("/data0/dianchen/forward_data", \
					# 	observation,
					# 	action,
					# 	next_ob)
					
					path_length += 1
					path_return += reward
					self.algo.pool.add_sample(observation,
					                     action,
					                     reward,
					                     terminal,
					                     False)
					if terminal or path_length >= self.algo.max_path_length:
					    self.algo.pool.add_sample(next_ob,
					                         np.zeros_like(action),
					                         np.zeros_like(reward),
					                         np.zeros_like(terminal),
					                         True)
					    observation = self.algo.training_env.reset()
					    self.algo.exploration_strategy.reset()
					    self.algo.es_path_returns.append(path_return)
					    path_length = 0
					    path_return = 0
					else:
					    observation = next_ob

					if self.algo.pool.size >= self.algo.min_pool_size:
					    for _ in range(self.algo.n_updates_per_time_step):
					        self._do_training(epoch * self.algo.epoch_length + t)
					itr += 1

				logger.log("Training finished. Time: {0}".format(time.time() -
                                                                 start_time))

				with self.algo._eval_then_training_mode():
					if self.algo.pool.size >= self.algo.min_pool_size:
						start_time = time.time()
						if self.algo.n_eval_samples > 0:
							# self.algo.evaluate(epoch, self.algo.es_path_returns)
							# self.algo.es_path_returns = []
							pass
						params = self.get_epoch_snapshot(epoch)
						logger.log(
							"Eval time: {0}".format(time.time() - start_time))
						logger.save_itr_params(epoch, params)
					logger.dump_tabular(with_prefix=False)
					logger.pop_prefix()
			self.algo._switch_to_eval_mode()
			self.algo.training_env.terminate()
			self.algo._shutdown_worker()
			return self.algo.last_statistics

	
	def _do_training(self, timestep):
		minibatch = self.algo.pool.random_batch(self.algo.batch_size)
		sampled_obs = minibatch["observations"]
		sampled_terminals = minibatch['terminals']
		sampled_next_obs = minibatch["next_observations"]
		sampled_actions = minibatch["actions"]
		sampled_rewards = minibatch['rewards']
		icm_feed_dict = self._update_feed_dict(sampled_rewards, sampled_obs, sampled_next_obs, sampled_actions)
		algo_ops = self.algo._get_training_ops()
		if timestep % self.algo_update_freq == 0:
			icm_ops = [self.icm_opt]
		else:
			icm_ops = []
		icm_results = self.sess.run([self.summary, self.internal_rewards] + icm_ops, feed_dict=icm_feed_dict)
		icm_summary = icm_results[0]
		internal_rewards = icm_results[1]
		# Add up internal and external rewards
		algo_feed_dict = self.algo._update_feed_dict(self.external_reward_weight * sampled_rewards + \
													(1. - self.external_reward_weight) * internal_rewards,
	       											 sampled_terminals,
	       											 sampled_obs,
	       											 sampled_actions,
	       											 sampled_next_obs)
		# If algo has summary, run it. 
		# TODO: Clean this code. It is a mess right now
		if self.algo.summary is not None:
			algo_ops = [self.algo.summary] + algo_ops
		algo_results = self.sess.run(algo_ops, feed_dict=algo_feed_dict)
		if self.algo.summary is not None:
			algo_summary = algo_results[0]
		if timestep % TENSORBOARD_PERIOD == 0:
			if self.algo.summary is not None:
				self.summary_writer.add_summary(algo_summary, timestep)
			self.summary_writer.add_summary(icm_summary, timestep)

	def _update_feed_dict(self, sampled_rewards, sampled_obs, sampled_next_obs, sampled_actions):
		return {
			self.s1: sampled_obs,
			self.s2: sampled_next_obs,
			self.asample: sampled_actions,
			self.external_rewards: sampled_rewards,
		}

	def get_epoch_snapshot(self, epoch):
		snapshot = self.algo.get_epoch_snapshot(epoch)
		snapshot['encoder'] = self._encoder
		snapshot['inverse_model'] = self._inverse_model
		snapshot['forward_model'] = self._forward_model
		return snapshot




