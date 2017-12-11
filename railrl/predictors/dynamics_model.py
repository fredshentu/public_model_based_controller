"""
Author: Dian Chen
"""

import abc

import tensorflow as tf
import numpy as np

from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
# from sandbox.rocky.tf.parametrized import Parametrized
from railrl.misc.rllab_util import get_action_dim
from railrl.core.neuralnet import NeuralNetwork
from railrl.predictors.state_action_network import StateActionNetwork
from railrl.predictors.state_network import StateNetwork
from railrl.core.tf_util import weight_variable, he_uniform_initializer, \
            xavier_uniform_initializer, mlp, linear, spatial_softmax, conv_network, deconv_network
from railrl.core.dcor import dCor, distance_matrix, double_center, U_center

from tensorflow.contrib.distributions import MultivariateNormalFull
# from tensorflow.image import resize_images
# from tf.contrib.losses import softmax_cross_entropy

import itertools
import math

# TODO: Use hierarchy here

class NoEncoder(StateNetwork, metaclass=abc.ABCMeta):
    def __init__(self, feature_dim, **kwargs):
        self.setup_serialization(locals())
        super().__init__(name_or_scope="encoder",
                         output_dim=feature_dim,
                         **kwargs)
    
    def get_features(self, observation):
        return observation
        
    @overrides
    def _create_network_internal(self, observation_input=None):
        assert observation_input is not None
        return observation_input



class FullyConnectedEncoder(StateNetwork, metaclass=abc.ABCMeta):
    def __init__(
            self, 
            feature_dim,
            hidden_sizes=(100,100),
            hidden_activation=tf.nn.relu,
            **kwargs
    ):
        self.setup_serialization(locals())
        self.feature_dim = feature_dim
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        super().__init__(name_or_scope="fc_encoder",
                         output_dim=feature_dim,
                         **kwargs)


    def get_features(self, observation):
        return self.sess.run(self.output, { self.observation_input: [observation]}, {})
    

    @overrides
    def _create_network_internal(self, observation_input=None):
        assert observation_input is not None
        observation_input = self._process_layer(observation_input,
                                                scope_name="observation_input")
        if len(self.hidden_sizes) > 0:
            with tf.variable_scope("mlp"):
                observation_output = mlp(
                    observation_input,
                    self.observation_dim,
                    self.hidden_sizes,
                    self.hidden_activation,
                    W_initializer=he_uniform_initializer(),
                    b_initializer=tf.constant_initializer(0.),
                    pre_nonlin_lambda=self._process_layer,
                )
            last_layer_size=self.hidden_sizes[-1]
        else:
            observation_output = observation_input
            last_layer_size=self.observation_dim
        with tf.variable_scope("output"):
            return linear(
                observation_output,
                last_layer_size,
                self.feature_dim,
                W_initializer=he_uniform_initializer(),
                b_initializer=tf.constant_initializer(0.),
            )

class OneLayer(StateNetwork, metaclass=abc.ABCMeta):
    def __init__(
            self, 
            feature_dim,
            hidden_sizes=(256,128,128,64),
            hidden_activation=tf.nn.relu,
            **kwargs
    ):
        self.setup_serialization(locals())
        self.feature_dim = feature_dim
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        # import pdb; pdb.set_trace()
        super().__init__(name_or_scope="one_layer",
                         output_dim=feature_dim,
                         **kwargs)


    def get_features(self, observation):
        return self.sess.run(self.output, { self.observation_input: [observation]}, {})
    

    @overrides
    def _create_network_internal(self, observation_input=None):
        assert observation_input is not None
        observation_input = self._process_layer(observation_input,
                                                scope_name="observation_input")
        
        # import pdb; pdb.set_trace()
        if len(self.hidden_sizes) > 0:
            with tf.variable_scope("mlp"):
                observation_output = mlp(
                    observation_input,
                    self.observation_dim,
                    self.hidden_sizes,
                    self.hidden_activation,
                    W_initializer=he_uniform_initializer(),
                    b_initializer=tf.constant_initializer(0.),
                    pre_nonlin_lambda=self._process_layer,
                )
            last_layer_size=self.hidden_sizes[-1]
        else:
            observation_output = observation_input
            last_layer_size=self.observation_dim
        with tf.variable_scope("output"):
            return linear(
                observation_output,
                last_layer_size,
                self.output_dim,
                W_initializer=he_uniform_initializer(),
                b_initializer=tf.constant_initializer(0.),
            )

class ConvEncoder(StateNetwork, metaclass=abc.ABCMeta):
    def __init__(
        self,
        feature_dim,
        input_shape,
        conv_filters=(32, 32, 32, 32, 32),
        conv_filter_sizes=((3,3),(3,3),(3,3),(3,3),(3,3)),
        conv_strides=(2, 2, 2, 2, 2),
        conv_pads=('SAME', 'SAME', 'SAME', 'SAME', 'SAME'),
        hidden_sizes=(256,),
        hidden_activation=tf.nn.relu,
        **kwargs
        ):

        self.setup_serialization(locals())
        self.input_shape = [None] + list(input_shape)
        self.feature_dim = feature_dim
        self.conv_filters = conv_filters
        self.conv_filter_sizes = conv_filter_sizes
        self.conv_strides = conv_strides
        self.conv_pads = conv_pads
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation

        super().__init__(name_or_scope="conv_encoder",
                         observation_dim=input_shape,
                         output_dim=feature_dim,
                         **kwargs)

    def get_features(self, observation):
        return self.sess.run(self.output, { self.observation_input: [observation]}, {})

    @overrides
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
                self.hidden_sizes,
                self.hidden_activation,
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
                self.hidden_sizes,
                self.hidden_activation,
                W_initializer=xavier_uniform_initializer(),
                b_initializer=tf.constant_initializer(0.),
                pre_nonlin_lambda=self._process_layer,
            )

        if len(self.hidden_sizes) > 0:
            last_size = self.hidden_sizes[-1]
        else:
            last_size = output_dim

        with tf.variable_scope("output"):
            return linear(
                observation_output,
                last_size,
                self.feature_dim,
                W_initializer=xavier_uniform_initializer(),
                b_initializer=tf.constant_initializer(0.),
            )


class ConvSpatialSoftMaxEncoder(ConvEncoder):
    def __init__(self, 
        feature_dim,
        input_shape,
        conv_filters=(32, 32, 32, 32, 32),
        conv_filter_sizes=((3,3),(3,3),(3,3),(3,3),(3,3)),
        conv_strides=(2, 2, 2, 2, 2),
        conv_pads=('SAME', 'SAME', 'SAME', 'SAME', 'SAME'),
        hidden_sizes=(256,),
        hidden_activation=tf.nn.relu,
        **kwargs
    ):
        super().__init__(
            feature_dim, 
            input_shape,
            conv_filters=conv_filters,
            conv_filter_sizes=conv_filter_sizes,
            conv_strides=conv_strides,
            conv_pads=conv_pads, **kwargs)

    @overrides
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
                self.hidden_sizes,
                self.hidden_activation,
                W_initializer=xavier_uniform_initializer(),
                b_initializer=tf.constant_initializer(0.),
                pre_nonlin_lambda=self._process_layer,
            )

            output, self.softmax, self.img_coord = spatial_softmax(observation_output, output_shape)

            return output / 42.0 - 0.5  # Normalize so that it is [-1, 1]

    

class DCorFeatureRegulator(NeuralNetwork, metaclass=abc.ABCMeta):
    def __init__(
        self,
        feature_dim,
        subfeature_dim,
        split_k=1,
        spatial_softmax=False,
        feature_input=None,
        output_activation=None,
        feature_converter=True,
        **kwargs
    ):

        assert subfeature_dim % split_k == 0

        self.setup_serialization(locals())
        super(DCorFeatureRegulator, self).__init__(name_or_scope="dcor_regulator", **kwargs)

        self.feature_dim = feature_dim
        self.subfeature_dim = subfeature_dim
        self.split_k = split_k
        self.output_activation = output_activation
        self.feature_converter = feature_converter
        self.spatial_softmax = spatial_softmax

        with tf.variable_scope(self.scope_name):
            if feature_input is None:
                if self.spatial_softmax:
                    feature_input = tf.placeholder(tf.float32,
                        [None] + list(feature_dim),
                        "_feature_input"
                    )
                else:
                    feature_input = tf.placeholder(tf.float32, 
                        [None, feature_dim],
                        "_feature_input")

        self.feature_input = feature_input

        self._create_network(feature_input=self.feature_input)


    @overrides
    def _create_network_internal(self, feature_input=None):
        # Build approx covariance function
        assert feature_input is not None

        self.feature_input = self._process_layer(feature_input, scope_name="feature_input")

        if not self.spatial_softmax and self.feature_converter:
            with tf.variable_scope("feature_converter"):
                self.W = weight_variable([self.feature_dim, self.subfeature_dim], initializer=xavier_uniform_initializer())
                feature_output = tf.matmul(feature_input, self.W)

        else:
            feature_output = feature_input

        self.feature_output = self._process_layer(feature_output, scope_name="subfeature")        

        if self.spatial_softmax:
            features = tf.unstack(self.feature_output, axis=1)
            num_split = len(features)

        else:
            num_split = int(self.subfeature_dim / self.split_k)
            features = tf.split(1,  num_split, self.feature_output)
        ## Only add up correlation
        
        loss = 0.0
        # self.losses = []
        # self.distance_matrices = []
        M = tf.cast(tf.shape(feature_output)[0], tf.float32)
        for i, j in itertools.product(range(num_split), range(num_split)):
            if i < j:
                # print (features[i].get_shape().as_list())
                # self.distance_matrices.append(distance_matrix(tf.expand_dims(features[i], 1)))
                # self.distance_matrices.append(distance_matrix(features[j]))
                # self.losses.append(dCor(features[i], features[j]))
                # loss += tf.square(self.losses[-1])
                loss += tf.square(dCor(features[i], features[j]))

        return loss / (M*(M-1)/2)


    @property
    @overrides
    def _input_name_to_values(self):
        return dict(
            feature_input=self.feature_input,
        )




class EntropyFeatureRegulator(NeuralNetwork, metaclass=abc.ABCMeta):
    def __init__(
        self, 
        feature_dim, 
        subfeature_dim, 
        feature_input=None, 
        mean_decay=0.99, 
        output_activation=tf.nn.tanh, 
        feature_converter=True,
        normalize_input=True,
        **kwargs
    ):
        self.setup_serialization(locals())
        super(EntropyFeatureRegulator, self).__init__(name_or_scope="entropy_regulator", **kwargs)

        self.feature_dim = feature_dim
        self.subfeature_dim = subfeature_dim
        self.mean_decay = mean_decay
        self.output_activation = output_activation
        self.feature_converter = feature_converter
        self.normalize_input = normalize_input

        with tf.variable_scope(self.scope_name):
             if feature_input is None:
                feature_input = tf.placeholder(tf.float32, 
                    [None, feature_dim],
                    "_feature_input")

        self.feature_input = feature_input

        self._create_network(feature_input=self.feature_input)


    @overrides
    def _create_network_internal(self, feature_input=None):
        # Build approx covariance function
        assert feature_input is not None

        self.feature_input = self._process_layer(feature_input, scope_name="feature_input")
        self.running_mean = tf.get_variable(
            dtype=tf.float32,
            shape=(self.subfeature_dim,),
            initializer=tf.constant_initializer(0.0),
            name="running_mean", trainable=False
        )

        with tf.variable_scope("feature_converter"):
            if self.feature_converter:
                self.W = weight_variable([self.feature_dim, self.subfeature_dim], initializer=xavier_uniform_initializer())
                feature_output = tf.matmul(feature_input, self.W)
                
            else:
                feature_output = feature_input

        self.feature_output = self._process_layer(feature_output, scope_name="subfeature")

        if self.normalize_input:
            # normed_feature_output = tf.nn.l2_normalize(self.feature_output - self.running_mean, 0)
            self.mean, self.var = tf.nn.moments(feature_output, axes=[0])
            self.normed_feature_output = (self.feature_output - self.mean) / (tf.sqrt(self.var) + 1e-12)
        else:
            self.normed_feature_output = self.feature_output

        with tf.variable_scope("approx_covariance"):
            M = tf.cast(tf.shape(feature_output)[0], tf.float32)
            self.covariance = 1. / M * (tf.matmul(tf.transpose(self.normed_feature_output), self.normed_feature_output))
            dist = MultivariateNormalFull([tf.zeros_like(self.mean)], [self.covariance])
            approx_entropy = dist.entropy()[0]

            self.update_mean_op = tf.assign(self.running_mean, 
                self.running_mean * self.mean_decay + tf.reduce_mean(feature_output, axis=0) * (1. - self.mean_decay)
            )

        return -approx_entropy


    @overrides
    def get_params_internal(self, **tags): 
        return super(EntropyFeatureRegulator, self).get_params_internal(**tags) + [self.running_mean]

    @property
    @overrides
    def _input_name_to_values(self):
        return dict(
            feature_input=self.feature_input,
        )


class Segmentor(NeuralNetwork, metaclass=abc.ABCMeta):
    def __init__(
        self,
        feature_dim,
        num_label=2,
        conv_filters=(1, 2, 2),
        conv_filter_sizes=((3,3),(3,3),(3,3)),
        conv_strides=(2, 2, 2),
        conv_pads=('SAME','SAME','SAME'),
        output_dim=64,
        upsample_dim=2048,
        hidden_activation=tf.nn.relu,
        feature_input=None,
        **kwargs
    ):
        self.setup_serialization(locals())
        super(Segmentor, self).__init__(name_or_scope="segmentor", **kwargs)

        # import pdb; pdb.set_trace()
        assert conv_filters[-1] == num_label and conv_filters[0] == 1

        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.upsample_dim = upsample_dim
        self.hidden_activation = hidden_activation

        self.conv_filters = conv_filters
        self.conv_filter_sizes = conv_filter_sizes
        self.conv_strides = conv_strides
        self.conv_pads = conv_pads

        with tf.variable_scope(self.scope_name):
            if feature_input is None:
                feature_input = tf.placeholder(
                    tf.float32,
                    [None, self.feature_dim], "_feature_input")
            
        self.feature_input = feature_input

        self._create_network(feature_input=self.feature_input)


    @overrides
    def _create_network_internal(self, feature_input):
        # reshape
        # assert 
        feature_output = mlp(
            feature_input,
            self.feature_dim,
            (self.output_dim,self.upsample_dim),
            self.hidden_activation,
            W_initializer=xavier_uniform_initializer(),
            b_initializer=tf.constant_initializer(0.),
            pre_nonlin_lambda=self._process_layer,
        )

        # side = int(math.sqrt(feature_output.get_shape().as_list()[1]))

        reshaped_feature_output = tf.reshape(feature_output, [tf.shape(feature_output)[0], 32, 32, 4])
        upsampled_output = tf.image.resize_images(reshaped_feature_output, [64, 64])
        return upsampled_output

        # return tf.nn.softmax(upsampled_output, dim=3)

        # reshaped_feature_output = tf.reshape(feature_output, [tf.shape(feature_output)[0], side, side, 1])
        # # import pdb; pdb.set_trace()

        # output, output_shape = deconv_network(
        #     reshaped_feature_output,
        #     [None, side, side, 1],
        #     self.conv_filters,
        #     self.conv_filter_sizes,
        #     self.conv_strides,
        #     self.conv_pads,
        #     (),
        #     self.hidden_activation,
        #     W_initializer=xavier_uniform_initializer(),
        #     b_initializer=tf.constant_initializer(0.),
        #     pre_nonlin_lambda=self._process_layer,
        # )

        # output = tf.nn.softmax(output, dim=3)
        # return output

    @property
    @overrides
    def _input_name_to_values(self):
        return dict(
            feature_input=self.feature_input,
        )



class InverseModel(NeuralNetwork, metaclass=abc.ABCMeta):
    def __init__(
            self, 
            feature_dim,
            hidden_sizes=(200,200),
            env_spec=None,
            action_dim=None,
            hidden_activation=tf.nn.relu,
            output_activation=tf.nn.tanh,
            feature_input1=None,
            feature_input2=None,
            **kwargs):
        self.setup_serialization(locals())
        super(InverseModel, self).__init__(name_or_scope="inverse_model", **kwargs)

        assert action_dim or env_spec
        
        self.action_dim = action_dim or env_spec.action_space.flat_dim
        self.feature_dim = feature_dim
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        with tf.variable_scope(self.scope_name):
            if feature_input1 is None:
                feature_input1 = tf.placeholder(
                    tf.float32,
                    [None, self.feature_dim], "_feature_input1")
            if feature_input2 is None:
                feature_input2 = tf.placeholder(
                    tf.float32,
                    [None, self.feature_dim], "_feature_input2")
        self.feature_input1 = feature_input1
        self.feature_input2 = feature_input2

        self._create_network(feature_input1=self.feature_input1, feature_input2=self.feature_input2)



    def get_action(self, feature1, feature2):
        return self.sess.run(self.output, 
                             { self.feature_input1: [feature_input1], self.feature_input2: [feature_input2] }, {})

    @overrides
    def _create_network_internal(self, feature_input1=None, feature_input2=None):
        assert feature_input1 is not None and feature_input2 is not None

        feature_input1 = self._process_layer(feature_input1, scope_name="feature_input1")
        feature_input2 = self._process_layer(feature_input2, scope_name="feature_input2")

        with tf.variable_scope("mlp"):
            embedded = tf.concat(1, [feature_input1, feature_input2])
            
            action_output = mlp(
                embedded,
                2 * self.feature_dim,
                self.hidden_sizes,
                self.hidden_activation,
                # W_initializer=lambda shape, dtype, partition_info: tf.truncated_normal(shape),
                b_initializer=tf.constant_initializer(0.),
                pre_nonlin_lambda=self._process_layer,
            )
        with tf.variable_scope("output"):
            if self.output_activation is not None:
                return self.output_activation(linear(
                    action_output,
                    self.hidden_sizes[-1],
                    self.action_dim,
                    # W_initializer=lambda shape, dtype, partition_info: tf.truncated_normal(shape),
                    b_initializer=tf.constant_initializer(0.),
                ))
            else:
                return linear(
                    action_output,
                    self.hidden_sizes[-1],
                    self.action_dim,
                    # W_initializer=lambda shape, dtype, partition_info: tf.truncated_normal(shape),
                    b_initializer=tf.constant_initializer(0.),
                )


    @property
    @overrides
    def _input_name_to_values(self):
        return dict(
            feature_input1=self.feature_input1,
            feature_input2=self.feature_input2,
        )


class ConditionalFeedForwardNet(NeuralNetwork, metaclass=abc.ABCMeta):
    def __init__(
            self, 
            feature_dim,
            hidden_sizes=(200,200,64),
            env_spec=None,
            action_dim=None,
            hidden_activation=tf.nn.relu,
            output_activation=tf.nn.tanh,
            feature_input=None,
            **kwargs):
        self.setup_serialization(locals())
        super(ConditionalFeedForwardNet, self).__init__(name_or_scope="conditional_feed_forward_net", **kwargs)

        assert action_dim or env_spec
        
        self.action_dim = action_dim or env_spec.action_space.flat_dim
        self.feature_dim = feature_dim
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        with tf.variable_scope(self.scope_name):
            if feature_input==None:
                feature_input = tf.placeholder(
                    tf.float32,
                    [None, self.feature_dim], "_feature_input")
                    
        self.feature_input = feature_input

        self._create_network(feature_input1=self.feature_input1, feature_input2=self.feature_input2)
    def get_action(self, feature_input):
        return self.sess.run(self.output, {self.feature_input: feature_input})
        
    @overrides
    def _create_network_internal(self, feature_input = None):
        assert feature_input is not None

        feature_input = self._process_layer(feature_input, scope_name="feature_input")

        with tf.variable_scope("mlp"):
            embedded = feature_input
            
            action_output = mlp(
                embedded,
                self.feature_dim,
                self.hidden_sizes,
                self.hidden_activation,
                # W_initializer=lambda shape, dtype, partition_info: tf.truncated_normal(shape),
                b_initializer=tf.constant_initializer(0.),
                pre_nonlin_lambda=self._process_layer,
            )

        outputs = []
        for i in range(self.action_dim):
            if i == 0:
                embedded = action_output
            else:
                embedded = tf.concat(1, [action_output, outputs[-1]])

            with tf.variable_scope("inverse_output%d" % i):
                linear_output = linear(
                    embedded,
                    embedded.get_shape().as_list()[1],
                    1,
                )

            if self.output_activation is not None:
                outputs.append(self.output_activation(linear_output))
            else:
                outputs.append(linear_output)

        return tf.concat(1, outputs)
    @property
    @overrides
    def _input_name_to_values(self):
        return dict(
            feature_input=self.feature_input,
        )


class ConditionalInverseModel(InverseModel):
    def __init__(
            self, 
            feature_dim,
            hidden_sizes=(200,200),
            env_spec=None,
            action_dim=None,
            hidden_activation=tf.nn.relu,
            output_activation=tf.nn.tanh,
            feature_input1=None,
            feature_input2=None,
            **kwargs):
        self.setup_serialization(locals())
        super(InverseModel, self).__init__(name_or_scope="inverse_model", **kwargs)

        assert action_dim or env_spec
        
        self.action_dim = action_dim or env_spec.action_space.flat_dim
        self.feature_dim = feature_dim
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        with tf.variable_scope(self.scope_name):
            if feature_input1 is None:
                feature_input1 = tf.placeholder(
                    tf.float32,
                    [None, self.feature_dim], "_feature_input1")
            if feature_input2 is None:
                feature_input2 = tf.placeholder(
                    tf.float32,
                    [None, self.feature_dim], "_feature_input2")
        self.feature_input1 = feature_input1
        self.feature_input2 = feature_input2

        self._create_network(feature_input1=self.feature_input1, feature_input2=self.feature_input2)

    @overrides
    def _create_network_internal(self, feature_input1=None, feature_input2=None):
        assert feature_input1 is not None and feature_input2 is not None

        feature_input1 = self._process_layer(feature_input1, scope_name="feature_input1")
        feature_input2 = self._process_layer(feature_input2, scope_name="feature_input2")

        with tf.variable_scope("mlp"):
            embedded = tf.concat(1, [feature_input1, feature_input2])
            
            action_output = mlp(
                embedded,
                2 * self.feature_dim,
                self.hidden_sizes,
                self.hidden_activation,
                # W_initializer=lambda shape, dtype, partition_info: tf.truncated_normal(shape),
                b_initializer=tf.constant_initializer(0.),
                pre_nonlin_lambda=self._process_layer,
            )

        outputs = []
        for i in range(self.action_dim):
            if i == 0:
                embedded = action_output
            else:
                embedded = tf.concat(1, [action_output, outputs[-1]])

            with tf.variable_scope("inverse_output%d" % i):
                linear_output = linear(
                    embedded,
                    embedded.get_shape().as_list()[1],
                    1,
                )

            if self.output_activation is not None:
                outputs.append(self.output_activation(linear_output))
            else:
                outputs.append(linear_output)

        return tf.concat(1, outputs)




class ForwardModel(NeuralNetwork, metaclass=abc.ABCMeta):
    def __init__(
            self,
            feature_dim,
            hidden_sizes=(300,300,200),
            env_spec=None,
            action_dim=None,
            hidden_activation=tf.nn.relu,
            feature_input=None,
            action_input=None,
            **kwargs):
        self.setup_serialization(locals())
        super(ForwardModel, self).__init__(name_or_scope="forward_model", **kwargs)
        assert action_dim or env_spec
        
        self.action_dim = action_dim or env_spec.action_space.flat_dim
        self.feature_dim = feature_dim
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        
        with tf.variable_scope(self.scope_name):
            if feature_input is None:
                feature_input = tf.placeholder(
                    tf.float32,
                    [None, self.feature_dim], "_feature_input")
            if action_input is None:
                action_input = tf.placeholder(
                    tf.float32,
                    [None, self.action_dim], "_action_input")
        
        self.feature_input = feature_input
        self.action_input = action_input
        self._create_network(feature_input=self.feature_input, action_input=self.action_input)



    @overrides
    def _create_network_internal(self, feature_input=None, action_input=None):
        assert feature_input is not None and action_input is not None

        feature_input = self._process_layer(feature_input, scope_name="feature_input")
        action_input = self._process_layer(action_input, scope_name="action_input")

        with tf.variable_scope("mlp"):
            embedded = tf.concat(1, [feature_input, action_input])
            embedded_dim = self.feature_dim + self.action_dim

            feature_output = mlp(
                embedded,
                embedded_dim,
                self.hidden_sizes,
                self.hidden_activation,
                # W_initializer=lambda shape, dtype, partition_info: tf.truncated_normal(shape),
                b_initializer=tf.constant_initializer(0.),
                pre_nonlin_lambda=self._process_layer,
            )

        with tf.variable_scope("output_linear"):
            return linear(
                feature_output,
                self.hidden_sizes[-1],
                self.feature_dim,
                # W_initializer=lambda shape, dtype, partition_info: tf.truncated_normal(shape),
                b_initializer=tf.constant_initializer(0.),
            )

    @property
    @overrides
    def _input_name_to_values(self):
        return dict(
            feature_input=self.feature_input,
            action_input=self.action_input,
        )

