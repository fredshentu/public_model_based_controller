"""
Author: Dian Chen
"""
from rllab.core.serializable import Serializable
from railrl.core import tf_util
from railrl.core.neuralnet import NeuralNetwork
from railrl.predictors.perceptron import Perceptron
from rllab.misc.overrides import overrides

class LSTM(NeuralNetwork):
    def __init__(
        self,
        name_or_scope,
        input_tensor,
        input_size,
        output_size,
        
    )