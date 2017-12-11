import abc
import tensorflow as tf

from railrl.core.neuralnet import NeuralNetwork
from railrl.core.tf_util import he_uniform_initializer, mlp, linear

class InverseDynamics(NeuralNetwork, metaclass=abc.ABCMeta):
	def __init__(
			self, 
			name_or_scope,
			action_dim,
			feature_dim,
			feature_input_t=None,
			feature_input_tp1=None,
			**kwargs)
	):
		self.setup_serialization(locals())
        super().__init__(name_or_scope=name_or_scope, **kwargs)
        self.action_dim = action_dim
        self.feature_dim = feature_dim

        with tf.variable_scope(self.scope_name):
            if feature_input_t is None:
                feature_input_t = tf.placeholder(
                    tf.float32,
                    [None, self.feature_dim],
                    "_f1")
            if feature_input_tp1 is None:
                feature_input_tp1 = tf.placeholder(
                    tf.float32,
                    [None, self.feature_dim],
                    "_f2")
        self.feature_input_t = feature_input_t
        self.feature_input_tp1 = feature_input_tp1
		self._create_network(feature_input_t=feature_input_t,
							 feature_input_tp1=feature_input_tp1)


	@property
    @overrides
    def _input_name_to_values(self):
        return dict(
        	feature_input_t=feature_input_t,
        	feature_input_tp1=feature_input_tp1
        )

    def __init__(self, )