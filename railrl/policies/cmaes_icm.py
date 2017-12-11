import tensorflow as tf

from rllab.policies.base import Policy
from rllab.spaces import Box
from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
from railrl.predictors.dynamics_model import FullyConnectedEncoder, InverseModel, ForwardModel
from pickle import dumps, loads
from copy import deepcopy
# import cma

class CMAESPolicy(Policy):
	"""
	A policy for effective control with learned dynamics model
	"""
	def __init__(
		self,
		env_spec,
		encoder: FullyConnectedEncoder,
		inverse_model: InverseModel,
		forward_model: ForwardModel,
		sess=None,
		sigma=0.5,
	):
		# assert isinstance(env_spec.action_space, Box)
		
		self.env_spec = env_spec
		self.action_dim = env_spec.action_space.flat_dim
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
		self.sess = sess or tf.get_default_session() or tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
		self.encoder = encoder
		self.inverse_model = inverse_model
		self.forward_model = forward_model
		self.inverse_model.sess = self.sess
		self.forward_model.sess = self.sess
		self.sigma = sigma
		

	@overrides
	def get_action(self, observation, env=None):
		import cma
		assert env is not None

		feature = self.encoder.get_features(observation)
		# env.render(close=True)
		snapshot = dumps(env)
		def forward_loss_func(action):
			old_data = env.wrapped_env.env.env.data
			old_qpos = old_data.qpos.flatten()
			old_qvel = old_data.qvel.flatten()
			# print (dir(old_data))
			# print ("old env")
			# print (old_data.qpos)
			# env_cpy = loads(snapshot)
			# env_cpy.reset()
			# env_cpy.wrapped_env.env.env.data = old_data
			# print ("new env")
			# print (env_cpy.wrapped_env.env.env.data.qpos)
			# env_cpy.wrapped_env.env.env.data = old_data
			next_o, r, d, env_info = env.step(action)
			next_feature = self.encoder.get_features(next_o)
			forward_loss = tf.reduce_sum(
				tf.square(self.forward_model.output - next_feature)
			)
			env.wrapped_env.env.env.set_state(old_qpos, old_qvel)
			env.wrapped_env.env._elapsed_steps -= 1
			# env.wrapped_env.env.env.data = old_data
			# Note it is to be negative to be maximized
			return -self.sess.run(
				forward_loss,
				feed_dict = {
					self.forward_model.feature_input: feature,
					self.forward_model.action_input: [action],
				}
			)

		res = cma.fmin(forward_loss_func, 
					  self.action_dim * [0.0], 
					  self.sigma,
					  options={'maxfevals': 10}
				)
		# print ("Best action found after %d, forward loss: %0.5f" % (res[2], -res[3]))
		return res[0], None


