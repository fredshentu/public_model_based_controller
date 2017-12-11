import os
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.uniform_control_policy import UniformControlPolicy
from rllab.envs.normalized_env import normalize
from railrl.algos.idle import IdleAlgo_TRPO as Idle
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv
import itertools

# import sys
# sys.path.append("/Users/dianchen/rail-rl")
from railrl.algos.icm_trpo_tf import ICM

stub(globals())

# Param ranges
seeds = range(5)

for seed in seeds:
    mdp = TfEnv(normalize(env=GymEnv('Box3dReach-v6',record_video=False, \
    log_dir='/tmp/gym_test',record_log=False)))
    
    policy = UniformControlPolicy(env.spec)

    baseline = LinearFeatureBaseline(
        mdp.spec,
    )

    batch_size = 5000
    algo = Idle(
        env=mdp, 
        policy=policy, 
        baseline=baseline,
    )

    algorithm = ICM(
        mdp,
        algo,
        "/home/dianchen/box3d/trpo_box3d_state_v6_tf_idle_%d"%seed,
        feature_dim=mdp.spec.observation_space.flat_dim,
        forward_weight=0.2,
        external_reward_weight=0.99,
        inverse_tanh=True,
        init_learning_rate=1e-4,
    )

    run_experiment_lite(
        algorithm.train(),
        exp_prefix='trpo_box3d_state_v6_tf_idle',
        n_parallel=1,
        snapshot_mode="gap",
        snapshot_gap=100,
        seed=seed,
        mode="local"
    )
