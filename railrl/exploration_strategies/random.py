from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.exploration_strategies.base import ExplorationStrategy
import numpy as np


class RandomStrategy(ExplorationStrategy, Serializable):
	"""
    This strategy always output a random action
    """
	def __init__(self, env_spec):
		assert isinstance(env_spec.action_space, Box)
		assert len(env_spec.action_space.shape) == 1
		Serializable.quick_init(self, locals())
		super().__init__()
		self._action_space = env_spec.action_space

	def get_action(self, t, observation, policy, **kwargs):
		return np.random.uniform(low=self._action_space.low, \
        						 high=self._action_space.high)
