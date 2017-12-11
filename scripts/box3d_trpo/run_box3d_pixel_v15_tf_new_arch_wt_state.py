import os
# from sandbox.rocky.tf.baselines.nn_baseline import NNBaseline
# from sandbox.rocky.tf.core.network import ConvNetwork
from sandbox.rocky.tf.core.network import ConvMergeNetwork
from sandbox.rocky.tf.baselines.nn_concat_baseline import NNConcatBaseline
from sandbox.rocky.tf.policies.gaussian_conv_feature_concat_policy import GaussianConvFeatureConcatPolicy
# from sandbox.rocky.tf.policies.gaussian_conv_feature_policy import GaussianConvFeaturePolicy
# from sandbox.rocky.tf.policies.gaussian_conv_policy import GaussianConvPolicy
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv
import itertools

from railrl.sampler.concat_sampler import ConcatBatchSampler

import tensorflow as tf

stub(globals())

# Params range
seeds = range(0, 2)

for seed in seeds:
    env = TfEnv(normalize(env=GymEnv('Box3dReachPixel-v15',record_video=False, \
    log_dir='/tmp/gym_test',record_log=False)))

    env_spec = env.spec
    policy_cnn = ConvMergeNetwork(
        name="conv_merge_policy_network",
        input_shape=env_spec.observation_space.shape,
        extra_input_shape=(8,),
        output_dim=env_spec.action_space.flat_dim,
        conv_filters=(32, 32, 32, 32, 32),
        conv_filter_sizes=((3,3),(3,3),(3,3),(3,3), (3,3)),
        conv_strides=(2, 2, 2, 2, 2),
        conv_pads=('SAME', 'SAME', 'SAME', 'SAME', 'SAME'),
        extra_hidden_sizes=(288,),
        hidden_sizes=(288,),
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=None,
    )
        
    baseline_cnn = ConvMergeNetwork(
        name="conv_merge_baseline_network",
        input_shape=env_spec.observation_space.shape,
        extra_input_shape=(8,),
        output_dim=env_spec.action_space.flat_dim,
        conv_filters=(32, 32, 32, 32, 32),
        conv_filter_sizes=((3,3),(3,3),(3,3),(3,3), (3,3)),
        conv_strides=(2, 2, 2, 2, 2),
        conv_pads=('SAME', 'SAME', 'SAME', 'SAME', 'SAME'),
        extra_hidden_sizes=(288,),
        hidden_sizes=(288,),
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=None,
    )

    policy = GaussianConvFeatureConcatPolicy(
        "conv_feature_policy",
        env_spec=env_spec,
        feature_network=policy_cnn,
        output_nonlinearity=tf.nn.tanh,
        hidden_sizes=(256,128),
        clip_action=False,
    )

    baseline = NNConcatBaseline(
        env_spec=env_spec,
        feature_network=baseline_cnn,
        hidden_sizes=(256,128),
        batch_size=256,
        hidden_nonlinearity=tf.nn.relu,
        init_lr=5e-4,
        n_itr=20,
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
        n_itr=1000,
        step_size=0.01,
        subsample_factor=1.0,
        sampler_cls=ConcatBatchSampler,
        optimizer_args={
            'num_slices' : 8,
        },
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix='trpo_box3d_pixel_v15_tf_new_arch_wt_state',
        n_parallel=12,
        snapshot_mode="gap",
        snapshot_gap=200,
        seed=seed,
        mode="local"
    )
