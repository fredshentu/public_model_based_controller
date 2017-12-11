"""
:author: Dian Chen
"""

import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

from rllab.spaces import Box, Discrete
from rllab.algos.base import RLAlgorithm
from rllab.misc.overrides import overrides
from rllab.misc import logger
from railrl.policies.nn_policy import NNPolicy
from railrl.algos.online_algorithm import OnlineAlgorithm


TENSORBOARD_PERIOD = 100

class ICM(RLAlgorithm):
	"""
	RL with inverse curiosity module
	"""
	def __init__(
			self,
			env,
			algo: OnlineAlgorithm,
			feature_dim=10,
			forward_weight=0.8,
			external_reward_weight=0.1,
			inverse_tanh=False,
			init_learning_rate=1e-4,
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
		self.external_reward_weight = external_reward_weight
		self.summary_writer = self.algo.summary_writer
		act_space = env.action_space
		obs_space = env.observation_space
		
		# Setup ICM models
		self.s1 = tf.placeholder(tf.float32, [None] + list(obs_space.shape))
		self.s2 = tf.placeholder(tf.float32, [None] + list(obs_space.shape))
		self.asample = tf.placeholder(tf.float32, [None, act_space.flat_dim])

		# Feature encoding
		if len(obs_space.shape) == 1:
			self.f1 = fc_encoder(self.s1, feature_dim)
			self.f2 = fc_encoder(self.s2, feature_dim, reuse=True)
		else:
			# TODO: implement conv encoder
			raise NotImplementedError("Currently only supports flat observation input!")

		# Inverse model: g(f1, f2) -> a_inv
		if isinstance(act_space, Box):
			if inverse_tanh:
				output_fn = tf.nn.tanh
			else:
				output_fn = None
			self.apred = cont_inverse_model(self.f1, self.f2, act_space.flat_dim, output_fn=output_fn)
		elif isinstance(obs_space, Discrete):
			# TODO: implement discrete inverse model
			raise NotImplementedError
		else:
			raise NotImplementedError

		# Forward model: f(f1, a) -> f2
		self.f2pred = forward_model(self.f1, self.asample)

		# Define losses
		if isinstance(act_space, Box):
			self.invloss = tf.nn.l2_loss(self.asample - self.apred)
		elif isinstance(act_space, Discrete):
			raise NotImplementedError
		else:
			raise NotImplementedError

		self.inverse_loss = tf.nn.l2_loss(self.asample - self.apred)
		self.forward_loss = tf.nn.l2_loss(self.f2 - self.f2pred)
		self.internal_rewards = tf.reduce_sum(tf.square(self.f2 - self.f2pred), axis=1)
		self.total_loss = forward_weight * self.forward_loss + \
						(1. - forward_weight) * self.inverse_loss

		self.icm_opt = tf.train.AdamOptimizer(init_learning_rate).\
						minimize(self.total_loss)
		
		self.sess = self.algo.sess

		# Setup summaries
		inverse_loss_summ = tf.summary.scalar("icm_inverse_loss", self.inverse_loss)
		forward_loss_summ = tf.summary.scalar("icm_forward_loss", self.forward_loss)
		total_loss_summ = tf.summary.scalar("icm_total_loss", self.total_loss)
		var_summ = []
		for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="icm"):
			var_summ.append(tf.summary.histogram(var.op.name, var))
		self.summary = tf.summary.merge([inverse_loss_summ, forward_loss_summ, total_loss_summ] + var_summ)

		# Initialize variables

		self.sess.run(tf.global_variables_initializer())



	@overrides
	def train(self):
		with self.sess.as_default():
			self.algo._init_training()
			self.algo._start_worker()
			self.algo._switch_to_training_mode()

			observation = self.algo.training_env.reset()
			itr = 0
			path_length = 0
			path_return = 0
			for epoch in range(self.algo.n_epochs):
				logger.push_prefix('Epoch #%d | ' % epoch)
				logger.log("Training started")
				start_time = time.time()
				for t in range(self.algo.epoch_length):
					with self.algo._eval_then_training_mode():
						action, _ = self.algo.policy.get_action(observation)

					if self.algo.render:
						self.training_env.render()
					next_ob, raw_reward, terminal, _ = self.algo.training_env.step(
						self.algo.process_action(action)
					)
					# Some envs return a Nx1 vector for the observation
					next_ob = next_ob.flatten()
					reward = raw_reward * self.algo.scale_reward
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
							self.algo.evaluate(epoch, self.algo.es_path_returns)
							self.algo.es_path_returns = []
						params = self.algo.get_epoch_snapshot(epoch)
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
		icm_feed_dict = self._update_feed_dict(sampled_obs, sampled_next_obs, sampled_actions)
		algo_ops = self.algo._get_training_ops()
		icm_ops = [self.icm_opt]
		icm_results = self.sess.run([self.summary, self.internal_rewards] + icm_ops, feed_dict=icm_feed_dict)
		icm_summary = icm_results[0]
		internal_rewards = icm_results[1]
		# Add up internal and external rewards
		algo_feed_dict = self.algo._update_feed_dict(self.external_reward_weight * sampled_rewards + internal_rewards,
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

	def _update_feed_dict(self, sampled_obs, sampled_next_obs, sampled_actions):
		return {
			self.s1: sampled_obs,
			self.s2: sampled_next_obs,
			self.asample: sampled_actions,
		}


def forward_model(f1, a, hidden_sizes=[20], activation_fn=tf.nn.relu):
	"""
	Forward model
	"""
	feature_dim = f1.get_shape().as_list()[1]
	with tf.variable_scope("icm/forward_model"):
		f = tf.concat(1, [f1, a])
		for hidden_size in hidden_sizes:
			f = tf.contrib.layers.fully_connected(f, hidden_size, activation_fn=activation_fn, weights_initializer=tf.contrib.layers.xavier_initializer())
		output = tf.contrib.layers.fully_connected(f, feature_dim, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
	return output


def cont_inverse_model(f1, f2, act_dim, hidden_sizes=[20], activation_fn=tf.nn.relu, output_fn=None):
	"""
	Inverse model with continuous action space
	"""
	with tf.variable_scope("icm/inverse_model"):
		g = tf.concat(1, [f1, f2])
		for hidden_size in hidden_sizes:
			g = tf.contrib.layers.fully_connected(g, hidden_size, activation_fn=activation_fn)
		output = tf.contrib.layers.fully_connected(g, int(act_dim), activation_fn=output_fn)
	return output


def fc_encoder(obs, feature_dim, hidden_sizes=[20], activation_fn=tf.nn.relu, reuse=False):
	"""
	Encoder with fully-connected architecture
	"""
	with tf.variable_scope("icm/encoder", reuse=reuse):
		for hidden_size in hidden_sizes:
			obs = tf.contrib.layers.fully_connected(obs, hidden_size, activation_fn=activation_fn, weights_initializer=tf.contrib.layers.xavier_initializer())
		output = tf.contrib.layers.fully_connected(obs, feature_dim, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
	return output







