"""
:author: Dian Chen
"""

import tensorflow as tf
from railrl.algos.online_algorithm import OnlineAlgorithm
from railrl.exploration_strategies.random import RandomStrategy
from rllab.misc.overrides import overrides
from rllab.policies.uniform_control_policy import UniformControlPolicy
from rllab.misc import logger
from rllab.algos.base import RLAlgorithm

from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler

class IdleAlgo(OnlineAlgorithm):
	"""
	An idle agent performing random actions all the time
	"""

	def __init__(self, env, tensorboard_path, **kwargs):
		exploration_strategy = RandomStrategy(env.spec)
		policy = UniformControlPolicy(env.spec)
		super().__init__(env, policy, exploration_strategy)

		self.summary_writer = tf.summary.FileWriter(tensorboard_path, graph=tf.get_default_graph())
		self.summary = None

	@overrides
	def _get_training_ops(self):
		return []

	@overrides
	def _update_feed_dict(self, rewards, terminals, obs, actions, next_obs):
			pass

	@overrides
	def _init_training(self):
		pass

	@overrides
	def evaluate(self, epoch, es_path_returns):
		logger.log("Skipping evaluation")
		return 

	@overrides
	def _switch_to_training_mode(self):
		pass

	@overrides
	def _switch_to_eval_mode(self):
		pass

	@overrides
	def get_epoch_snapshot(self, epoch):
		return dict(
			env=self.training_env
		)

class IdleAlgo_TRPO(RLAlgorithm):
	def __init__(self, 
			env,
			policy,
			baseline,
			n_itr=1000, 
			scope=None,
			start_itr=0,
			batch_size=5000,
			max_path_length=500,
			discount=0.99,
			gae_lambda=1,
			plot=False,
			pause_for_plot=False,
			center_adv=True,
			positive_adv=False,
			store_paths=False,
			whole_paths=True,
			fixed_horizon=False,
			sampler_cls=None,
			sampler_args=None,
			force_batch_sampler=False,
			**kwargs
		):
		super(IdleAlgo_TRPO, self).__init__()
		self.env = env
		self.policy = policy
		self.baseline = baseline
		self.scope = scope
		self.n_itr = n_itr
		self.start_itr = start_itr
		self.batch_size = batch_size
		self.max_path_length = max_path_length
		self.discount = discount
		self.gae_lambda = gae_lambda
		self.plot = plot
		self.pause_for_plot = pause_for_plot
		self.center_adv = center_adv
		self.positive_adv = positive_adv
		self.store_paths = store_paths
		self.whole_paths = whole_paths
		self.fixed_horizon = fixed_horizon
		if sampler_cls is None:
			if self.policy.vectorized and not force_batch_sampler:
				sampler_cls = VectorizedSampler
			else:
				sampler_cls = BatchSampler
		sampler_args = dict()
		self.sampler = sampler_cls(self, **sampler_args)


	@overrides
	def optimize_policy(self, itr, samples_data):
		logger.log("Skipping optimization")

	@overrides
	def get_itr_snapshot(self, itr, samples_data):
		return dict(env=self.env, policy=self.policy)

	@overrides
	def log_diagnostics(self, paths):
		pass

	def start_worker(self):
		self.sampler.start_worker()

	def shutdown_worker(self):
		self.sampler.shutdown_worker()

	def obtain_samples(self, itr):
		return self.sampler.obtain_samples(itr)

	def process_samples(self, itr, paths):
		return self.sampler.process_samples(itr, paths)

