import os
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv

from railrl.algos.ddpg import DDPG
from railrl.policies.nn_policy import FeedForwardPolicy
from railrl.qfunctions.nn_qfunction import FeedForwardCritic
from rllab.exploration_strategies.ou_strategy import OUStrategy
from railrl.core.tf_util import BatchNormConfig

import itertools
import argparse

import tensorflow as tf

stub(globals())


def main():

    parser = argparse.ArgumentParser()
    # Hyperparameters
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--policy_initlr', type=float, default=1e-4)
    parser.add_argument('--qf_initlr', type=float, default=1e-3)
    
    parser.add_argument('--qf_decay', type=float, default=.0)
    parser.add_argument('--qf_soft_tau', type=float, default=1e-3)

    # Exploration hyperparameters
    parser.add_argument('--ou_theta', type=float, default=0.15)
    parser.add_argument('--ou_sigma', type=float, default=0.3)   


    parser.add_argument('--tfboard_path', type=str, default='/tmp/tfboard')
    parser.add_argument('--gpu_ratio', type=float, default=1.0)

    args = parser.parse_args()

    env = TfEnv(normalize(env=GymEnv('Box3dReach-v11',record_video=False, \
    log_dir='/tmp/gym_test',record_log=False)))

    name = 'ddpg-state-v11-plr{0}-qlr{1}-tau{2}-qfdecay{3}-ou_theta{4}-ou_sigma{5}'.format(
        args.policy_initlr,
        args.qf_initlr,
        args.qf_soft_tau,
        args.qf_decay,
        args.ou_theta,
        args.ou_sigma
    )
    
    es = OUStrategy(env_spec=env.spec, theta=args.ou_theta, sigma=args.ou_sigma)

    policy = FeedForwardPolicy(
        name_or_scope="actor",
        observation_hidden_sizes=(400,300),
        env_spec=env.spec,
    )

    qf = FeedForwardCritic(
        name_or_scope="critic",
        env_spec=env.spec,
        embedded_hidden_sizes=(100,),
        observation_hidden_sizes=(100,),
    )

    algo = DDPG(
        env=env, 
        exploration_strategy=es,
        policy=policy,
        qf=qf,
        tensorboard_path=os.path.join(args.tfboard_path, name, '_%d'%args.seed),
        qf_learning_rate=args.qf_initlr,
        policy_learning_rate=args.policy_initlr,
        soft_target_tau=args.qf_soft_tau,
        gpu_ratio=args.gpu_ratio,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix=name,
        n_parallel=1,
        snapshot_mode="last",
        seed=args.seed,
        mode="local"
    )

if __name__ == '__main__':
    main()