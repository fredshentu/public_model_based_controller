import os
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv
import itertools

# import sys
# sys.path.append("/Users/dianchen/rail-rl")
from railrl.algos.icm_trpo_tf import ICM
import tensorflow as tf

stub(globals())

# Param ranges
seeds = range(5)

for seed in seeds:
    mdp = TfEnv(normalize(env=GymEnv('Box3dReach-v12',record_video=False, \
    log_dir='/tmp/gym_test',record_log=False)))
    
    policy = GaussianMLPPolicy(
        "new_mlp_policy",
        env_spec=mdp.spec,
        hidden_sizes=(64, 64, 32),
        output_nonlinearity=tf.nn.tanh,
        clip_action=False,
    )

    baseline = LinearFeatureBaseline(
        mdp.spec,
    )

    batch_size = 50000
    algo = TRPO(
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        whole_paths=True,
        max_path_length=1000,
        n_itr=1000,
        step_size=0.01,
        subsample_factor=1.0,
        sampler_cls=BatchSampler,
    )

    algorithm = ICM(
        mdp,
        algo,
        "/media/4tb/box3d/trpo_box3d_state_v12_tf_icm_frozen_fw0.1_frozen_%d"%seed,
        feature_dim=mdp.spec.observation_space.flat_dim,
        pretrained_icm=True,
        pretrained_icm_path="/home/fred/rllab/data/local/trpo-state-v12-tf-icm-fw0.1-initlr-0.001/trpo-state-v12-tf-icm-fw0.1-initlr-0.001_2017_07_16_22_12_20_0001/itr_1000.pkl",
        freeze_icm=True,
        external_reward_weight=0.0,
    )

    run_experiment_lite(
        algorithm.train(),
        exp_prefix='trpo_box3d_state_v12_tf_icm_frozen_fw0.1',
        n_parallel=8,
        snapshot_mode="gap",
        snapshot_gap=200,
        seed=seed,
        mode="local"
    )
