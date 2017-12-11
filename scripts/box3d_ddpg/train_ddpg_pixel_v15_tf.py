import os
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv

from railrl.algos.ddpg import DDPG
from railrl.policies.nn_policy import ConvNNPolicy
from railrl.qfunctions.nn_qfunction import ConvNNCritic
from rllab.exploration_strategies.ou_strategy import OUStrategy
from railrl.core.tf_util import BatchNormConfig

import itertools
import argparse

import tensorflow as tf

def main():

    parser = argparse.ArgumentParser()
    # Hyperparameters
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--policy_initlr', type=float, default=1e-4)
    parser.add_argument('--qf_initlr', type=float, default=1e-3)
    
    parser.add_argument('--qf_decay', type=float, default=0.01)
    parser.add_argument('--qf_soft_tau', type=float, default=1e-3)

    # Exploration hyperparameters
    parser.add_argument('--ou_theta', type=float, default=0.15)
    parser.add_argument('--ou_sigma', type=float, default=0.3)   


    parser.add_argument('--tfboard_path', type=str, default='/tmp/tfboard')
    parser.add_argument('--gpu_ratio', type=float, default=0.95)

    args = parser.parse_args()

    env = TfEnv(normalize(env=GymEnv('Box3dReachPixel-v15',record_video=False, \
    log_dir='/tmp/gym_test',record_log=False)))

    name = 'ddpg-pixel-v15-plr{0}-qlr{1}-tau{2}-qfdecay{3}'.format(
        args.policy_initlr,
        args.qf_initlr,
        args.qf_soft_tau,
        args.qf_decay
    )
    
    es = OUStrategy(env_spec=env.spec, theta=args.ou_theta, sigma=args.ou_sigma)
    # import pdb; pdb.set_trace()

    qf = ConvNNCritic(
        name_or_scope="critic",
        input_shape=env.observation_space.shape,
        env_spec=env.spec,
        conv_filters=(32, 32, 32, 32, 32),
        conv_filter_sizes=((3,3),(3,3),(3,3),(3,3),(3,3)),
        conv_strides=(2, 2, 2, 2, 2),
        conv_pads=('SAME', 'SAME', 'SAME', 'SAME', 'SAME'),
        observation_hidden_sizes=(256,),
        embedded_hidden_sizes=(256,),
        hidden_nonlinearity=tf.nn.relu,
    )


    policy = ConvNNPolicy(
        name_or_scope="actor",
        input_shape=env.observation_space.shape,
        env_spec=env.spec,
        conv_filters=(32, 32, 32, 32, 32),
        conv_filter_sizes=((3,3),(3,3),(3,3),(3,3),(3,3)),
        conv_strides=(2, 2, 2, 2, 2),
        conv_pads=('SAME', 'SAME', 'SAME', 'SAME', 'SAME'),
        observation_hidden_sizes=(256,128),
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.nn.tanh,
    )


    algo = DDPG(
        env=env, 
        exploration_strategy=es,
        policy=policy,
        qf=qf,
        tensorboard_path=os.path.join(args.tfboard_path, name + '_%d'%args.seed),
        replay_pool_size=100000,
        obs_dtype='uint8',
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