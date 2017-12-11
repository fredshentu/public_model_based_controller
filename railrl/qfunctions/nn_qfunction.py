import abc
import tensorflow as tf
import numpy as np

from railrl.predictors.state_action_network import StateActionNetwork
from railrl.core.tf_util import he_uniform_initializer, xavier_uniform_initializer, mlp, conv_network, linear


class NNQFunction(StateActionNetwork, metaclass=abc.ABCMeta):
    def __init__(
            self,
            name_or_scope,
            **kwargs
    ):
        self.setup_serialization(locals())
        super().__init__(name_or_scope=name_or_scope, output_dim=1, **kwargs)


class FeedForwardCritic(NNQFunction):
    def __init__(
            self,
            name_or_scope,
            hidden_W_init=None,
            hidden_b_init=None,
            output_W_init=None,
            output_b_init=None,
            embedded_hidden_sizes=(100,),
            observation_hidden_sizes=(100,),
            hidden_nonlinearity=tf.nn.relu,
            **kwargs
    ):
        self.setup_serialization(locals())
        self.hidden_W_init = hidden_W_init or he_uniform_initializer()
        self.hidden_b_init = hidden_b_init or tf.constant_initializer(0.)
        self.output_W_init = output_W_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.output_b_init = output_b_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.embedded_hidden_sizes = embedded_hidden_sizes
        self.observation_hidden_sizes = observation_hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        super().__init__(name_or_scope=name_or_scope, **kwargs)

    def _create_network_internal(self, observation_input, action_input):
        observation_input = self._process_layer(observation_input,
                                                scope_name="observation_input")
        action_input = self._process_layer(action_input,
                                           scope_name="action_input")
        with tf.variable_scope("observation_mlp"):
            observation_output = mlp(
                observation_input,
                self.observation_dim,
                self.observation_hidden_sizes,
                self.hidden_nonlinearity,
                W_initializer=self.hidden_W_init,
                b_initializer=self.hidden_b_init,
                pre_nonlin_lambda=self._process_layer,
            )
            observation_output = self._process_layer(
                observation_output,
                scope_name="observation_output"
            )
        embedded = tf.concat(1, [observation_output, action_input])
        embedded_dim = self.action_dim + self.observation_hidden_sizes[-1]
        with tf.variable_scope("fusion_mlp"):
            fused_output = mlp(
                embedded,
                embedded_dim,
                self.embedded_hidden_sizes,
                self.hidden_nonlinearity,
                W_initializer=self.hidden_W_init,
                b_initializer=self.hidden_b_init,
                pre_nonlin_lambda=self._process_layer,
            )
            fused_output = self._process_layer(fused_output)

        with tf.variable_scope("output_linear"):
            return linear(
                fused_output,
                self.embedded_hidden_sizes[-1],
                1,
                W_initializer=self.output_W_init,
                b_initializer=self.output_b_init,
            )

class ConvNNCritic(NNQFunction):
    def __init__(self,
                 name_or_scope,
                 input_shape,
                 conv_filters=(32, 32, 32, 32, 32),
                 conv_filter_sizes=((3,3),(3,3),(3,3),(3,3),(3,3)),
                 conv_strides=(2, 2, 2, 2, 2),
                 conv_pads=('SAME', 'SAME', 'SAME', 'SAME', 'SAME'),
                 observation_hidden_sizes=(256,),
                 embedded_hidden_sizes=(256,),
                 hidden_W_init=None,
                 hidden_b_init=None,
                 output_W_init=None,
                 output_b_init=None,
                 hidden_nonlinearity=tf.nn.relu,
                 **kwargs
    ):
        self.setup_serialization(locals())
        self.input_shape = input_shape
        self.hidden_W_init = hidden_W_init or xavier_uniform_initializer()
        self.hidden_b_init = hidden_b_init or tf.constant_initializer(0.)
        self.output_W_init = output_W_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.output_b_init = output_b_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.conv_filters = conv_filters
        self.conv_filter_sizes = conv_filter_sizes
        self.conv_strides = conv_strides
        self.conv_pads = conv_pads
        self.embedded_hidden_sizes = embedded_hidden_sizes
        self.observation_hidden_sizes = observation_hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        super().__init__(name_or_scope=name_or_scope, observation_dim=input_shape, **kwargs)


    def _create_network_internal(self, observation_input=None, action_input=None):
        assert observation_input is not None and action_input is not None
        observation_input = self._process_layer(observation_input,
                                                scope_name="observation_input")
        action_input = self._process_layer(action_input,
                                           scope_name="action_input")
        with tf.variable_scope("conv_network"):
            observation_output, output_shape = conv_network(
                observation_input,
                self.input_shape,
                self.conv_filters,
                self.conv_filter_sizes,
                self.conv_strides,
                self.conv_pads,
                self.observation_hidden_sizes,
                self.hidden_nonlinearity,
                W_initializer=xavier_uniform_initializer(),
                b_initializer=tf.constant_initializer(0.),
                pre_nonlin_lambda=self._process_layer,
            )

        output_dim = np.prod(output_shape[1:])
        observation_output = tf.contrib.layers.flatten(observation_output, [-1, output_dim])

        with tf.variable_scope("mlp"):
            observation_output = mlp(
                observation_output,
                output_dim,
                self.observation_hidden_sizes,
                self.hidden_nonlinearity,
                W_initializer=xavier_uniform_initializer(),
                b_initializer=tf.constant_initializer(0.),
                pre_nonlin_lambda=self._process_layer,
            )

        embedded = tf.concat(1, [observation_output, action_input])
        embedded_dim = self.action_dim + self.observation_hidden_sizes[-1]
        with tf.variable_scope("fusion_mlp"):
            fused_output = mlp(
                embedded,
                embedded_dim,
                self.embedded_hidden_sizes,
                self.hidden_nonlinearity,
                W_initializer=self.hidden_W_init,
                b_initializer=self.hidden_b_init,
                pre_nonlin_lambda=self._process_layer,
            )
            fused_output = self._process_layer(fused_output)

        with tf.variable_scope("output"):
            return linear(
                observation_output,
                self.embedded_hidden_sizes[-1],
                1,
                W_initializer=xavier_uniform_initializer(),
                b_initializer=tf.constant_initializer(0.),
            )

        # observation_output = tf.reshape(observation_input, [-1, 56448])
        # with tf.variable_scope("output"):
        #     return linear(
        #         observation_output,
        #         56448,
        #         1,
        #         W_initializer=xavier_uniform_initializer(),
        #         b_initializer=tf.constant_initializer(0.),
        #     )
