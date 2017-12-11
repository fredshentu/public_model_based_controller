"""
:author: Dian Chen
"""

import time

import numpy as np
import tensorflow as tf

from copy import copy

from rllab.spaces import Box, Discrete
from rllab.algos.base import RLAlgorithm
from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.algos.trpo import TRPO
from railrl.predictors.dynamics_model import NoEncoder, FullyConnectedEncoder, InverseModel, ForwardModel
from railrl.misc.icm_util import save_data
from railrl.data_management.simple_replay_pool import TRPOReplayPool
from railrl.sampler.base import BatchSampler

TENSORBOARD_PERIOD = 500



import numpy as np
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algos import util
import rllab.misc.logger as logger

class ICM(RLAlgorithm):
	"""
	RL with inverse curiosity module
	"""
	def __init__(
			self,
			env,
			trpo: TRPO,
			tensorboard_path,
			no_encoder=False,
			feature_dim=10,
			forward_weight=0.8,
			external_reward_weight=0.01,
			inverse_tanh=True,
			init_learning_rate=1e-4,
			icm_batch_size=128,
			replay_pool_size=1000000,
			min_pool_size=1000,
			n_updates_per_sample=500,
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
		self.trpo = trpo
		self.trpo.sampler = BatchSampler(self.trpo)
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
		self.sess = tf.get_default_session() or tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
		self.external_reward_weight = external_reward_weight
		self.summary_writer = tf.summary.FileWriter(tensorboard_path, graph=tf.get_default_graph())
		self.n_updates_per_sample = n_updates_per_sample
		self.icm_batch_size = icm_batch_size
		act_space = env.action_space
		obs_space = env.observation_space

		self.pool = TRPOReplayPool(replay_pool_size, obs_space.flat_dim, act_space.flat_dim)
		self.min_pool_size = min_pool_size
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
		external_rewards = tf.summary.scalar("mean_external_rewards", self.mean_external_rewards)
		var_summ = []
		# for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
		# 	var_summ.append(tf.summary.histogram(var.op.name, var))
		self.summary = tf.summary.merge([inverse_loss_summ, forward_loss_summ, total_loss_summ] + var_summ)
		# Initialize variables
		
		self.sess.run(tf.initialize_all_variables())


	@overrides
	def train(self):
		with self.sess.as_default():
			self.trpo.start_worker()
			self.trpo.init_opt()
			for itr in range(self.trpo.current_itr, self.trpo.n_itr):
				paths = self.trpo.sampler.obtain_samples(itr)
				modified_paths = self.process_paths(itr, paths)
				samples_data = self.trpo.sampler.process_samples(itr, modified_paths)
				
				if self.pool.size >= self.min_pool_size:
					logger.log("ICM Training started")
					start_time = time.time()
					for _ in range(self.n_updates_per_sample):
						self.train_icm(_ + itr * self.n_updates_per_sample)

					logger.log("ICM Training finished. Time: {0}".format(time.time() - start_time))
				
				for path in samples_data['paths']:
					path_len = len(path['rewards'])
					for i in range(path_len):
						obs = path['observations'][i]
						# print (obs)
						act = path['actions'][i]
						term = (i == path_len - 1)
						rew = 0.0
						self.pool.add_sample(obs, act, rew, term)

				self.trpo.log_diagnostics(paths)
				self.trpo.optimize_policy(itr, samples_data)
				params = self.trpo.get_itr_snapshot(itr, samples_data)
				params['encoder'] = self._encoder
				params['inverse_model'] = self._inverse_model
				params['forward_model'] = self._forward_model
				logger.save_itr_params(itr, params)
				logger.dump_tabular(with_prefix=False)

			self.trpo.shutdown_worker()


	def train_icm(self, timestep):
		batch = self.pool.random_batch(self.icm_batch_size)
		obs = batch['observations']
		next_obs = batch['next_observations']
		acts = batch['actions']
		rewards = batch['rewards']
		feed_dict = self._update_feed_dict(rewards, obs, next_obs, acts)
		ops = [self.summary, self.icm_opt]
		results = self.sess.run(ops, feed_dict=feed_dict)
		if timestep % TENSORBOARD_PERIOD == 0:
			self.summary_writer.add_summary(results[0], timestep)

	def process_paths(self, itr, paths):
		modified_paths = copy(paths)
		if itr == 0:
			for path in modified_paths:
				path['t_rewards'] = path["rewards"]
		else:
			for path in modified_paths:
				obs = path['observations'][:-1]
				acts = path['actions'][:-1]
				next_obs = path['observations'][1:]
				internal_rewards = self.sess.run(self.internal_rewards, feed_dict={
					self.s1: obs,
					self.s2: next_obs,
					self.asample: acts
				})
				internal_rewards = np.append([0.0], internal_rewards)
				path['t_rewards'] = self.external_reward_weight * path['rewards'] \
									+ (1. - self.external_reward_weight) * internal_rewards
		return modified_paths



	def _update_feed_dict(self, sampled_rewards, sampled_obs, sampled_next_obs, sampled_actions):
		return {
			self.s1: sampled_obs,
			self.s2: sampled_next_obs,
			self.asample: sampled_actions,
			self.external_rewards: sampled_rewards,
		}





