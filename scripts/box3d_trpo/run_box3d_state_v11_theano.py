import os
os.environ["THEANO_FLAGS"] = "device=cuda"

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import normalize
from rllab.algos.trpo import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.envs.gym_env import GymEnv
import itertools

# import sys
# sys.path.append("/Users/dianchen/rail-rl")
stub(globals())

# Param ranges
seeds = range(5)

for seed in seeds:
    mdp = normalize(env=GymEnv('Box3dReach-v11',record_video=False, \
    log_dir='/tmp/gym_test',record_log=False))
    
    policy = GaussianMLPPolicy(
        "mlp_policy",
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
        max_path_length=200,
        n_itr=1000,
        step_size=0.01,
        subsample_factor=1.0,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix='trpo_box3d_state_v11_theano',
        n_parallel=8,
        snapshot_mode="gap",
        snapshot_gap=200,
        seed=seed,
        mode="local"
    )
