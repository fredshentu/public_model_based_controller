import abc

import tensorflow as tf
import numpy as np

from railrl.core.tf_util import he_uniform_initializer, xavier_uniform_initializer, mlp, conv_network, linear
from railrl.misc.rllab_util import get_action_dim
from railrl.predictors.state_network import StateNetwork
from rllab.misc.overrides import overrides
from rllab.policies.base import Policy


class NNPolicy(StateNetwork, Policy, metaclass=abc.ABCMeta):
    def __init__(
            self,
            name_or_scope,
            **kwargs
    ):
        self.setup_serialization(locals())
        action_dim = get_action_dim(**kwargs)
        # Copy dict to not affect kwargs, which is used by Serialization
        new_kwargs = dict(**kwargs)
        if "action_dim" in new_kwargs:
            new_kwargs.pop("action_dim")
        super(NNPolicy, self).__init__(name_or_scope=name_or_scope,
                                       output_dim=action_dim,
                                       **new_kwargs)

    def get_action(self, observation):
        return self.sess.run(self.output,
                             {self.observation_input: [observation]}), {}


class FeedForwardPolicy(NNPolicy):
    def __init__(
            self,
            name_or_scope,
            observation_hidden_sizes=(300, 300),
            hidden_W_init=None,
            hidden_b_init=None,
            output_W_init=None,
            output_b_init=None,
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh,
            **kwargs
    ):
        self.setup_serialization(locals())
        self.observation_hidden_sizes = observation_hidden_sizes
        self.hidden_W_init = hidden_W_init or he_uniform_initializer()
        self.hidden_b_init = hidden_b_init or tf.constant_initializer(0.)
        self.output_W_init = output_W_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.output_b_init = output_b_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity
        super().__init__(name_or_scope=name_or_scope, **kwargs)

    def _create_network_internal(self, observation_input=None):
        assert observation_input is not None
        observation_input = self._process_layer(observation_input,
                                                scope_name="observation_input")
        with tf.variable_scope("mlp"):
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
            scope_name="output_preactivations",
        )
        with tf.variable_scope("output"):
            return self.output_nonlinearity(linear(
                observation_output,
                self.observation_hidden_sizes[-1],
                self.output_dim,
                W_initializer=self.output_W_init,
                b_initializer=self.output_b_init,
            ))

class ConvNNPolicy(NNPolicy):
    def __init__(self,
                 name_or_scope,
                 input_shape,
                 conv_filters=(32, 32, 32, 32, 32),
                 conv_filter_sizes=((3,3),(3,3),(3,3),(3,3),(3,3)),
                 conv_strides=(2, 2, 2, 2, 2),
                 conv_pads=('SAME', 'SAME', 'SAME', 'SAME', 'SAME'),
                 observation_hidden_sizes=(256,128),
                 hidden_W_init=None,
                 hidden_b_init=None,
                 output_W_init=None,
                 output_b_init=None,
                 hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=tf.nn.tanh,
                 **kwargs
    ):
        self.setup_serialization(locals())
        self.input_shape = input_shape
        self.observation_hidden_sizes = observation_hidden_sizes
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
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity
        self.reshaped_observation_feature = None
        super().__init__(name_or_scope=name_or_scope, observation_dim=input_shape, **kwargs)


    def _create_network_internal(self, observation_input=None):
        assert observation_input is not None
        observation_input = self._process_layer(observation_input,
                                                scope_name="observation_input")

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
        observation_output = tf.reshape(observation_output, [-1, output_dim], name="reshape")
        # observation_output = tf.contrib.layers.flatten(observation_output, [-1, output_dim])
        self.reshaped_observation_feature = observation_output
        # tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, observation_output)


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

        with tf.variable_scope("output"):
            return self.output_nonlinearity(linear(
                observation_output,
                self.observation_hidden_sizes[-1],
                self.output_dim,
                W_initializer=xavier_uniform_initializer(),
                b_initializer=tf.constant_initializer(0.),
            ))

    
    def get_params_internal_pixel(self):

        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                      self.full_scope_name)
        variables.append(self.reshaped_observation_feature)

        return variables