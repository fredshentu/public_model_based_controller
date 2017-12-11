from rllab.envs.normalized_env import normalize
from railrl.predictors.dynamics_model import ConvEncoder, InverseModel, ForwardModel
from railrl.predictors.mlp import Mlp
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv

import argparse
import numpy as np
import tensorflow as tf

ENV_NAME = "Box3dReachPixel-v3"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, 
                         help='path for icm training data')
    parser.add_argument('--maxiter', type=int, default=100000)

    args = parser.parse_args()

    env = TfEnv(normalize(env=GymEnv(ENV_NAME, record_video=False, \
    log_dir='/tmp/gym_test',record_log=False)))

    env_spec = env.spec

    # Build ICM
    encoder = ConvEncoder(
        feature_dim=200,
        input_shape=env_spec.observation_space.shape,
        conv_filters=(32, 32, 32, 32, 32),
        conv_filter_sizes=((3,3),(3,3),(3,3),(3,3),(3,3)),
        conv_strides=(2, 2, 2, 2, 2),
        conv_pads=('SAME', 'SAME', 'SAME', 'SAME', 'SAME'),
        hidden_sizes=(256,),
        hidden_activation=tf.nn.relu,
    )

    fc_net = Mlp(
        "fc-net",
        encoder.output,
        256,
        env_spec.action_space.flat_dim,
        (128,64)
    )

    