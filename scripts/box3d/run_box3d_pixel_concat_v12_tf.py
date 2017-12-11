import os
from sandbox.rocky.tf.baselines.nn_concat_baseline import NNConcatBaseline
from sandbox.rocky.tf.core.network import ConvMergeNetwork
from sandbox.rocky.tf.policies.gaussian_conv_feature_concat_policy import GaussianConvFeatureConcatPolicy
from sandbox.rocky.tf.policies.gaussian_conv_policy import GaussianConvPolicy
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.samplers.vectorized_concat_sampler import VectorizedConcatSampler
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv
import itertools

import tensorflow as tf

stub(globals())

# Params range
seeds = range(0, 5)

for seed in seeds:
    env = TfEnv(normalize(env=GymEnv('Box3dReachPixel-v12',record_video=False, \
    log_dir='/tmp/gym_test',record_log=False)))
    
    env_spec = env.spec
    # cnn = ConvNetwork(
    #     name="conv_feature_network",
    #     input_shape=env_spec.observation_space.shape,
    #     output_dim=env_spec.action_space.flat_dim,
    #     conv_filters=(32, 32, 32, 32, 32),
    #     conv_filter_sizes=((3,3),(3,3),(3,3),(3,3), (3,3)),
    #     conv_strides=(2, 2, 2, 2, 2),
    #     conv_pads=('SAME', 'SAME', 'SAME', 'SAME', 'SAME'),
    #     hidden_sizes=(256,),
    #     hidden_nonlinearity=tf.nn.relu,
    #     output_nonlinearity=None,
    # )

    merge_cnn = ConvMergeNetwork(
        name="conv_merge_feature_network",
        input_shape=env_spec.observation_space.shape,
        extra_input_shape=(8,),
        output_dim=env_spec.action_space.flat_dim,
        conv_filters=(32, 32, 32, 32, 32),
        conv_filter_sizes=((3,3),(3,3),(3,3),(3,3), (3,3)),
        conv_strides=(2, 2, 2, 2, 2),
        conv_pads=('SAME', 'SAME', 'SAME', 'SAME', 'SAME'),
        hidden_sizes=(256,256),
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=None,
    )

    policy = GaussianConvFeatureConcatPolicy(
        "conv_feature_policy",
        env_spec=env_spec,
        feature_network=merge_cnn,
        hidden_sizes=(128,64),
        output_nonlinearity=tf.nn.tanh,
    )

    baseline = NNConcatBaseline(
        env_spec=env_spec,
        feature_network=merge_cnn,
        hidden_sizes=(128,64),
        hidden_nonlinearity=tf.nn.relu,
        init_lr=0.001,
        n_itr=5,
        train_feature_network=False,
    )

    batch_size = 2400
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        whole_paths=True,
        max_path_length=200,
        n_itr=2000,
        step_size=0.01,
        subsample_factor=1.0,
        sampler_cls=VectorizedConcatSampler
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix='trpo_box3d_pixel_concat_v12_tf',
        n_parallel=1,
        snapshot_mode="gap",
        snapshot_gap=200,
        seed=seed,
        mode="local"
    )