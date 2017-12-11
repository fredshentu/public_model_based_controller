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
    mdp = TfEnv(normalize(env=GymEnv('Box3dReach-v4',record_video=False, \
    log_dir='/tmp/gym_test',record_log=False)))
    
    policy = GaussianMLPPolicy(
        "mlp_policy",
        env_spec=mdp.spec,
        hidden_sizes=(64, 64, 32),
        output_nonlinearity=tf.nn.tanh,
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
        max_path_length=200,
        n_itr=1000,
        step_size=0.01,
        subsample_factor=1.0,
        sampler_cls=BatchSampler,
    )

    algorithm = ICM(
        mdp,
        algo,
        "/z/dianchen/box3d/trpo_box3d_state_v4_tf_icm_ext0.9995_%d"%seed,
        feature_dim=mdp.spec.observation_space.flat_dim,
        forward_weight=0.1,
        external_reward_weight=0.9995,
        replay_pool_size=500000,
        init_learning_rate=5e-4,
        n_updates_per_iter=1000
    )

    run_experiment_lite(
        algorithm.train(),
        exp_prefix='trpo_box3d_state_v4_tf_icm_ext0.9995',
        n_parallel=6,
        snapshot_mode="gap",
        snapshot_gap=100,
        seed=seed,
        mode="local"
    )
