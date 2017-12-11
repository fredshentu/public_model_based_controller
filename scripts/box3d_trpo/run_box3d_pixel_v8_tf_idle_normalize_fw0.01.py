import os
from sandbox.rocky.tf.baselines.nn_baseline import NNBaseline
from sandbox.rocky.tf.core.network import ConvNetwork
from sandbox.rocky.tf.policies.gaussian_conv_feature_policy import GaussianConvFeaturePolicy
from sandbox.rocky.tf.policies.gaussian_conv_policy import GaussianConvPolicy
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.normalized_env import normalize
from railrl.algos.idle import IdleAlgo_TRPO as Idle
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv

from railrl.predictors.dynamics_model import ConvEncoder, InverseModel, ForwardModel
from railrl.algos.icm_trpo_tf import ICM

import itertools

import tensorflow as tf

stub(globals())

# Params range
seeds = range(0, 3)

for seed in seeds:
    env = TfEnv(normalize(env=GymEnv('Box3dReachPixel-v8',record_video=False, \
    log_dir='/tmp/gym_test',record_log=False)))
    
    env_spec = env.spec
    cnn = ConvNetwork(
        name="conv_feature_network",
        input_shape=env_spec.observation_space.shape,
        output_dim=env_spec.action_space.flat_dim,
        conv_filters=(32, 32, 32, 32, 32),
        conv_filter_sizes=((3,3),(3,3),(3,3),(3,3), (3,3)),
        conv_strides=(2, 2, 2, 2, 2),
        conv_pads=('SAME', 'SAME', 'SAME', 'SAME', 'SAME'),
        hidden_sizes=(256,),
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=None,
    )

    policy = GaussianConvFeaturePolicy(
        "conv_feature_policy",
        env_spec=env_spec,
        feature_network=cnn,
        hidden_sizes=(128,64),
        output_nonlinearity=tf.nn.tanh,
    )

    baseline = NNBaseline(
        env_spec=env_spec,
        feature_network=cnn,
        hidden_sizes=(128,64),
        hidden_nonlinearity=tf.nn.relu,
        init_lr=0.001,
        n_itr=5,
    )

    batch_size = 2400
    idle = Idle(
        env=env, 
        policy=policy, 
        baseline=baseline,
    )

    icm = ICM(
        env,
        idle,
        '/home/fshentu/box3d/trpo_box3d_pixel_v8_tf_idle_normalize_fw_0.01_%d'%seed,
        feature_dim=256,
        forward_weight=0.01,
        inverse_tanh=True,
        init_learning_rate=1e-4,
        icm_batch_size=128,
        replay_pool_size=10000,
        n_updates_per_iter=200,
        obs_dtype='uint8',
        normalize_input=True,
    )

    run_experiment_lite(
        icm.train(),
        exp_prefix='trpo_box3d_pixel_v8_tf_idle_normalize',
        n_parallel=1,
        snapshot_mode="gap",
        snapshot_gap=200,
        seed=seed,
        mode="local"
    )
