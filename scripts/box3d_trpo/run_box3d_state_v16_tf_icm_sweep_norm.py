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

import argparse

# import sys
# sys.path.append("/Users/dianchen/rail-rl")
from railrl.algos.icm_trpo_tf import ICM
import tensorflow as tf

stub(globals())

def main():

    parser = argparse.ArgumentParser()
    # Hyperparameters
    parser.add_argument('--fw_ratio', type=float, default=0.1)
    parser.add_argument('--init_lr', type=float, default=5e-4)

    parser.add_argument('--tfboard_path', type=str, default='/tmp/tfboard')
    parser.add_argument('--gpu_ratio', type=float, default=0.99)

    args = parser.parse_args()

    # Param ranges
    seeds = range(2)

    for seed in seeds:
        mdp = TfEnv(normalize(env=GymEnv('Box3dReach-v16',record_video=False, \
        log_dir='/tmp/gym_test',record_log=False), normalize_obs=True))

        name = 'trpo-state-v16-tf-icm-fw{}-initlr-{}-norm'.format(
            args.fw_ratio,
            args.init_lr
        )
        
        policy = GaussianMLPPolicy(
            "mlp_policy",
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
            args.tfboard_path + "/%s_%d"%(name, seed),
            feature_dim=mdp.spec.observation_space.flat_dim,
            forward_weight=args.fw_ratio,
            external_reward_weight=0.0,
            replay_pool_size=1000000,
            init_learning_rate=args.init_lr,
            n_updates_per_iter=1000,
        )

        run_experiment_lite(
            algorithm.train(),
            exp_prefix=name,
            n_parallel=8,
            snapshot_mode="gap",
            snapshot_gap=200,
            seed=seed,
            mode="local"
        )

if __name__ == '__main__':
    main()

