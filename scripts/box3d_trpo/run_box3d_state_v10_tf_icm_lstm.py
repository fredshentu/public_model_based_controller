import os
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv
import itertools

# import sys
# sys.path.append("/Users/dianchen/rail-rl")
from railrl.algos.icm_trpo_tf_box3d import ICM
import tensorflow as tf

stub(globals())

# Param ranges
seeds = range(5)

for seed in seeds:
    mdp = TfEnv(normalize(env=GymEnv('Box3dReach-v10',record_video=False, \
    log_dir='/tmp/gym_test',record_log=False)))
    
    # policy = GaussianMLPPolicy(
    #     "mlp_policy",
    #     env_spec=mdp.spec,
    #     hidden_sizes=(64, 32),
    #     output_nonlinearity=tf.nn.tanh,
    #     clip_action=True,
    # )

    policy = GaussianLSTMPolicy(
        "lstm_policy",
        env_spec=mdp.spec,
        hidden_dim=64,
        output_nonlinearity=tf.tanh,
    )

    baseline = LinearFeatureBaseline(
        mdp.spec,
    )

    batch_size = 5000
    algo = TRPO(
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        whole_paths=True,
        max_path_length=500,
        n_itr=2000,
        step_size=0.01,
        subsample_factor=1.0,
    )

    algorithm = ICM(
        mdp,
        algo,
        "/data0/dianchen/box3d/trpo_box3d_state_v10_tf_icm_lstm_%d"%seed,
        feature_dim=mdp.spec.observation_space.flat_dim,
        forward_weight=0.2,
        external_reward_weight=0.99,
        inverse_tanh=True,
        init_learning_rate=1e-4,
    )

    run_experiment_lite(
        algorithm.train(),
        exp_prefix='trpo_box3d_state_v10_tf_icm_lstm',
        n_parallel=1,
        snapshot_mode="gap",
        snapshot_gap=100,
        seed=seed,
        mode="local"
    )
