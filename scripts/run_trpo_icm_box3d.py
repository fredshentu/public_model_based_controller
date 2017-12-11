import os
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
#from rllab.envs.mujoco.gather.swimmer_gather_env import SwimmerGatherEnv
os.environ["THEANO_FLAGS"] = "device=cpu"
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import NormalizedEnv
from rllab.algos.trpo import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.envs.gym_env import GymEnv
from railrl.algos.icm_trpo import ICM
import itertools

stub(globals())

# Param ranges
seeds = range(2)
# SwimmerGather hierarchical task

for seed in seeds:
    mdp = NormalizedEnv(env=GymEnv('Box3dReach-v2',record_video=False, \
	log_dir='/tmp/gym_test',record_log=False))
    policy = GaussianMLPPolicy(
        env_spec=mdp.spec,
        hidden_sizes=(64, 32),
    )

    baseline = LinearFeatureBaseline(
        mdp.spec,
    )

    batch_size = 50000
    trpo = TRPO(
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        whole_paths=True,
        max_path_length=500,
        n_itr=10000,
        step_size=0.01,
        subsample_factor=1.0,
    )

    algorithm = ICM(
        mdp, 
        algo,
        variant['tensorboard'],
        no_encoder=False,
        feature_dim=env.spec.observation_space.flat_dim,
        forward_weight=0.9,
        external_reward_weight=0.9,
        inverse_tanh=True,
        init_learning_rate=1e-3,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="trpo_icm",
        n_parallel=4,
        snapshot_mode="last",
        seed=seed,
        mode="local"
    )
