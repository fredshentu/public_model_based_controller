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
from railrl.algos.icm_trpo_tf import ICM

import itertools

import tensorflow as tf

stub(globals())

# Params range
seeds = range(0, 5)

for seed in seeds:
    env = TfEnv(normalize(env=GymEnv('Box3dReachPixel-v11',record_video=False, \
    log_dir='/tmp/gym_test',record_log=False)))
    
    env_spec = env.spec
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
        clip_action=False,
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

    batch_size = 9600
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        whole_paths=True,
        max_path_length=1000,
        n_itr=2000,
        step_size=0.01,
        subsample_factor=0.2,
        sampler_cls=BatchSampler,
        optimizer_args={
            'num_slices' : 4,
        }
    )

    icm = ICM(
        env,
        algo,
        '/z/dianchen/box3d/trpo_box3d_pixel_v11_tf_icm_pretrained_cnn_norew_fw0.001_%d'%seed,
        forward_weight=0.001,
        external_reward_weight=0.0,
        init_learning_rate=1e-4,
        forward_cos=True,
        replay_pool_size=100000,
        n_updates_per_iter=500,
        normalize_input=True,
        obs_dtype='uint8',
        pretrained_icm=True,
        pretrained_icm_path='/z/dianchen/tfmodel_box3d/icm_supervised_box3dpixel_v11_box_dense_2e3_fw_0.001_lr_5e-4.pkl',
    )

    run_experiment_lite(
        icm.train(),
        exp_prefix='trpo_box3d_pixel_v11_tf_icm_pretrained_cnn_norew_fw0.001',
        n_parallel=12,
        snapshot_mode="gap",
        snapshot_gap=200,
        seed=seed,
        mode="local"
    )
