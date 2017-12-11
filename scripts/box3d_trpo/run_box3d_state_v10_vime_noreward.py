import os
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.mujoco.gather.swimmer_gather_env import SwimmerGatherEnv
os.environ["THEANO_FLAGS"] = "device=cuda"

from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import NormalizedEnv

from sandbox.vime.algos.trpo_expl import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.envs.gym_env import GymEnv
import itertools

stub(globals())

# Param ranges
seeds = range(5)
# SwimmerGather hierarchical task


for seed in seeds:
    eta = 0.0001
    mdp = NormalizedEnv(env=GymEnv('Box3dReach-v10',record_video=False, \
        log_dir='/tmp/gym_test',record_log=False))
    policy = GaussianMLPPolicy(
        env_spec=mdp.spec,
        hidden_sizes=(64, 64, 32),
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
        max_path_length=500,
        n_itr=1000,
        step_size=0.01,
        eta=eta,
        snn_n_samples=10,
        subsample_factor=1.0,
        use_replay_pool=True,
        use_kl_ratio=True,
        use_kl_ratio_q=True,
        n_itr_update=1,
        kl_batch_size=1,
        normalize_reward=False,
        replay_pool_size=1000000,
        n_updates_per_sample=5000,
        second_order_update=True,
        unn_n_hidden=[64,64],
        unn_layers_type=[1, 1, 1],
        unn_learning_rate=0.0001,
        no_reward=True,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="trpo-box3d-state-v10-no-reward",
        n_parallel=8,
        snapshot_mode="last",
        seed=seed,
        mode="local",
        script="sandbox/vime/experiments/run_experiment_lite.py",
    )
