import os
from sandbox.rocky.tf.baselines.nn_baseline import NNBaseline
from sandbox.rocky.tf.core.network import ConvNetwork
from sandbox.rocky.tf.policies.gaussian_conv_feature_policy import GaussianConvFeaturePolicy
from sandbox.rocky.tf.policies.gaussian_conv_policy import GaussianConvPolicy
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv
import itertools

import tensorflow as tf

stub(globals())

# Params range
seeds = range(0, 5)

for seed in seeds:
    env = TfEnv(normalize(env=GymEnv('Box3dReachPixel-v11',record_video=False, \
    log_dir='/tmp/gym_test',record_log=False)))
    
    env_spec = env.spec
    with tf.device("/gpu:0"):
        policy_cnn = ConvNetwork(
            name="policy_conv_network",
            input_shape=env_spec.observation_space.shape,
            output_dim=env_spec.action_space.flat_dim,
            conv_filters=(64, 64, 64, 32),
            conv_filter_sizes=((5,5),(3,3),(3,3),(3,3)),
            conv_strides=(3, 3, 3, 2),
            conv_pads=('SAME', 'SAME', 'SAME', 'SAME'),
            hidden_sizes=(256,),
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=None,
        )
    with tf.device(":/gpu:1"):
        baseline_cnn = ConvNetwork(
            name="baseline_conv_network",
            input_shape=env_spec.observation_space.shape,
            output_dim=env_spec.action_space.flat_dim,
            conv_filters=(64, 64, 64, 32),
            conv_filter_sizes=((5,5),(3,3),(3,3),(3,3)),
            conv_strides=(3, 3, 3, 2),
            conv_pads=('SAME', 'SAME', 'SAME', 'SAME'),
            hidden_sizes=(256,),
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=None,
        )

    policy = GaussianConvFeaturePolicy(
        "conv_feature_policy",
        env_spec=env_spec,
        feature_network=policy_cnn,
        hidden_sizes=(128,64),
        output_nonlinearity=tf.nn.tanh,
    )

    baseline = NNBaseline(
        env_spec=env_spec,
        feature_network=baseline_cnn,
        hidden_sizes=(128,64),
        hidden_nonlinearity=tf.nn.relu,
        init_lr=0.001,
        n_itr=5,
        train_feature_network=True,
    )

    batch_size = 5000
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        whole_paths=True,
        max_path_length=1000,
        n_itr=1400,
        step_size=0.01,
        subsample_factor=1.0,
        sampler_cls=BatchSampler,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix='trpo_box3d_pixel_v11_tf_2_cnn',
        n_parallel=1,
        snapshot_mode="gap",
        snapshot_gap=200,
        seed=seed,
        mode="local"
    )