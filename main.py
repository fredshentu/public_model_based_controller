"""Test different rl algorithms."""
import argparse
import copy

import tensorflow as tf

from railrl.launchers.algo_launchers import (
    icm_launcher,
    my_ddpg_launcher,
    naf_launcher,
    random_action_launcher,
    shane_ddpg_launcher,
    rllab_vpg_launcher,
    rllab_trpo_launcher,
    rllab_ddpg_launcher,
    tf_trpo_launcher,
    quadratic_ddpg_launcher,
    run_experiment,
    oat_qddpg_launcher,
)
from railrl.launchers.launcher_util import get_env_settings
from railrl.misc import hyperparameter as hp

BATCH_SIZE = 128
N_EPOCHS = 100
EPOCH_LENGTH = 10000
EVAL_SAMPLES = 10000
DISCOUNT = 0.99
QF_LEARNING_RATE = 1e-3
POLICY_LEARNING_RATE = 1e-4
BATCH_LEARNING_RATE = 1e-2
SOFT_TARGET_TAU = 1e-2
REPLAY_POOL_SIZE = 1000000
MIN_POOL_SIZE = 10000
SCALE_REWARD = 1.0
QF_WEIGHT_DECAY = 0.0001
MAX_PATH_LENGTH = 1000
N_UPDATES_PER_TIME_STEP = 5
BATCH_NORM_PARAMS = None  # None = off, {} = default params

# Sweep settings
SWEEP_N_EPOCHS = 20
SWEEP_EPOCH_LENGTH = 10000
SWEEP_EVAL_SAMPLES = 10000
SWEEP_MIN_POOL_SIZE = 10000

# Fast settings
FAST_N_EPOCHS = 3
FAST_EPOCH_LENGTH = 3
FAST_EVAL_SAMPLES = 3
FAST_MIN_POOL_SIZE = 5
FAST_MAX_PATH_LENGTH = 5

NUM_SEEDS_PER_CONFIG = 5
NUM_HYPERPARAMETER_CONFIGS = 50


def get_launch_settings_list_from_args(args):
    render = args.render

    def get_launch_settings(algo_name):
        """
        Return a dictionary of the form
        {
            'algo_params': algo_params to pass to run_algorithm
            'variant': variant to pass to run_algorithm
        }
        :param algo_name: Name of the algorithm to run.
        :return:
        """
        sweeper = hp.RandomHyperparameterSweeper()
        algo_params = {}
        if algo_name == 'ddpg':
            sweeper = hp.RandomHyperparameterSweeper([
                hp.LogFloatParam("qf_learning_rate", 1e-5, 1e-2),
                hp.LogFloatParam("policy_learning_rate", 1e-6, 1e-3),
                hp.LogFloatParam("scale_reward", 10.0, 0.001),
                hp.LogFloatParam("soft_target_tau", 1e-5, 1e-2),
            ])
            algo_params = get_ddpg_params()
            algo_params['render'] = render
            algorithm_launcher = my_ddpg_launcher
            variant = {
                'Algorithm': 'DDPG',
                'qf_params': dict(
                    embedded_hidden_sizes=(400,300),
                    observation_hidden_sizes=(200,200),
                    hidden_nonlinearity=tf.nn.relu,
                ),
                'policy_params': dict(
                    observation_hidden_sizes=(200, 200),
                    hidden_nonlinearity=tf.nn.tanh,
                    output_nonlinearity=tf.nn.tanh,
                )
            }
        elif algo_name == 'shane-ddpg':
            sweeper = hp.RandomHyperparameterSweeper([
                hp.LogFloatParam("soft_target_tau", 0.005, 0.1),
                hp.LogFloatParam("scale_reward", 10.0, 0.01),
                hp.LogFloatParam("qf_weight_decay", 1e-7, 1e-1),
            ])
            algo_params = get_ddpg_params()
            if algo_params['min_pool_size'] <= algo_params['batch_size']:
                algo_params['min_pool_size'] = algo_params['batch_size'] + 1
            algorithm_launcher = shane_ddpg_launcher
            variant = {'Algorithm': 'Shane-DDPG', 'policy_params': dict(
                hidden_sizes=(100, 100),
                hidden_nonlinearity=tf.nn.relu,
                output_nonlinearity=tf.nn.tanh,
            ), 'qf_params': dict(
                hidden_sizes=(100, 100)
            )}
        elif algo_name == 'qddpg':
            sweeper = hp.RandomHyperparameterSweeper([
                hp.LogFloatParam("soft_target_tau", 0.005, 0.1),
                hp.LogFloatParam("scale_reward", 10.0, 0.01),
                hp.LogFloatParam("qf_weight_decay", 1e-7, 1e-1),
                hp.LogFloatParam("qf_learning_rate", 1e-6, 1e-2),
                hp.LogFloatParam("policy_learning_rate", 1e-6, 1e-2),
            ])
            algo_params = get_ddpg_params()
            algorithm_launcher = quadratic_ddpg_launcher
            variant = {
                'Algorithm': 'QuadraticDDPG',
                'qf_params': dict(),
                'policy_params': dict(
                    observation_hidden_sizes=(100, 100),
                    hidden_nonlinearity=tf.nn.relu,
                    output_nonlinearity=tf.nn.tanh,
                )
            }
        elif algo_name == 'oat':
            algo_params = get_ddpg_params()
            algorithm_launcher = oat_qddpg_launcher
            variant = {
                'Algorithm': 'QuadraticOptimalActionTargetDDPG',
                'qf_params': dict(),
                'policy_params': dict(
                    observation_hidden_sizes=(100, 100),
                    hidden_nonlinearity=tf.nn.relu,
                    output_nonlinearity=tf.nn.tanh,
                )
            }
        elif algo_name == 'naf':
            sweeper = hp.RandomHyperparameterSweeper([
                hp.LogFloatParam("qf_learning_rate", 1e-5, 1e-2),
                hp.LogFloatParam("scale_reward", 10.0, 0.001),
                hp.LogFloatParam("soft_target_tau", 1e-6, 1e-1),
                hp.LogFloatParam("qf_weight_decay", 1e-7, 1e-1),
            ])
            algo_params = get_my_naf_params()
            algo_params['render'] = render
            algorithm_launcher = naf_launcher
            variant = {
                'Algorithm': 'NAF',
                'exploration_strategy_params': {
                    'sigma': 0.15
                },
            }
        elif algo_name == 'random':
            algorithm_launcher = random_action_launcher
            variant = {'Algorithm': 'Random'}
        elif algo_name == 'idle':
            # TODO: implement idle launcher
            algorithm_launcher = None
            variant = {'Algorithm': 'Idle'}
        elif algo_name == 'rl-vpg':
            algorithm_launcher = rllab_vpg_launcher
            algo_params = dict(
                batch_size=BATCH_SIZE,
                max_path_length=MAX_PATH_LENGTH,
                n_itr=N_EPOCHS,
                discount=DISCOUNT,
                optimizer_args=dict(
                    tf_optimizer_args=dict(
                        learning_rate=BATCH_LEARNING_RATE,
                    )
                ),
            )
            variant = {'Algorithm': 'rllab-VPG'}
        elif algo_name == 'rl-trpo':
            algorithm_launcher = rllab_trpo_launcher
            algo_params = dict(
                batch_size=BATCH_SIZE,
                max_path_length=MAX_PATH_LENGTH,
                n_itr=N_EPOCHS,
                discount=DISCOUNT,
                step_size=BATCH_LEARNING_RATE,
            )
            variant = {'Algorithm': 'rllab-TRPO'}
        elif algo_name == 'tf-trpo':
            algorithm_launcher = tf_trpo_launcher
            algo_params = dict(
                batch_size=BATCH_SIZE,
                max_path_length=MAX_PATH_LENGTH,
                n_itr=N_EPOCHS,
                discount=DISCOUNT,
                step_size=BATCH_LEARNING_RATE,
            )
            variant = {'Algorithm': 'tf-TRPO'}
        elif algo_name == 'rl-ddpg':
            algorithm_launcher = rllab_ddpg_launcher
            algo_params = get_ddpg_params()
            if algo_params['min_pool_size'] <= algo_params['batch_size']:
                algo_params['min_pool_size'] = algo_params['batch_size'] + 1
            variant = {'Algorithm': 'rllab-DDPG'}
        else:
            raise Exception("Algo name not recognized: " + algo_name)

        # bn_sweeper = hp.RandomHyperparameterSweeper([
        #     hp.EnumParam("decay", [0.9, 0.99, 0.999, 0.9999]),
        #     hp.LogFloatParam("epsilon", 1e-3, 1e-7),
        #     hp.EnumParam("enable_offset", [True, False]),
        #     hp.EnumParam("enable_scale", [True, False]),
        # ])
        bn_sweeper = None
        return {
            'sweeper': sweeper,
            'batch_norm_sweeper': bn_sweeper,
            'variant': variant,
            'algo_params': algo_params,
            'algorithm_launcher': algorithm_launcher,
            'batch_norm_params': BATCH_NORM_PARAMS
        }

    return [get_launch_settings(algo_name) for algo_name in args.algo]


def get_ddpg_params():
    return dict(
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        epoch_length=EPOCH_LENGTH,
        eval_samples=EVAL_SAMPLES,
        discount=DISCOUNT,
        policy_learning_rate=POLICY_LEARNING_RATE,
        qf_learning_rate=QF_LEARNING_RATE,
        soft_target_tau=SOFT_TARGET_TAU,
        replay_pool_size=REPLAY_POOL_SIZE,
        min_pool_size=MIN_POOL_SIZE,
        scale_reward=SCALE_REWARD,
        max_path_length=MAX_PATH_LENGTH,
        qf_weight_decay=QF_WEIGHT_DECAY,
    )


def get_my_naf_params():
    return dict(
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        epoch_length=EPOCH_LENGTH,
        eval_samples=EVAL_SAMPLES,
        discount=DISCOUNT,
        qf_learning_rate=QF_LEARNING_RATE,
        soft_target_tau=SOFT_TARGET_TAU,
        replay_pool_size=REPLAY_POOL_SIZE,
        min_pool_size=MIN_POOL_SIZE,
        scale_reward=SCALE_REWARD,
        max_path_length=MAX_PATH_LENGTH,
        qf_weight_decay=QF_WEIGHT_DECAY,
        n_updates_per_time_step=N_UPDATES_PER_TIME_STEP,
    )


def run_algorithm(
        launch_settings,
        env_params,
        exp_prefix,
        seed,
        tensorboard_path,
        icm, 
        exp_id=1,
        **kwargs):
    """
    Launch an algorithm
    :param launch_settings: See get_launch_settings_list_from_args
    :param env_params: See get_env_settings
    :param exp_prefix: Experiment prefix
    :param seed: Experiment seed
    :param exp_id: Experiment ID # to identify it later (e.g. for plotting data)
    :param kwargs: Other kwargs to pass to run_experiment_lite
    :return:
    """
    variant = launch_settings['variant']
    variant['env_params'] = env_params
    variant['algo_params'] = launch_settings['algo_params']
    variant['batch_norm_params'] = launch_settings['batch_norm_params']
    variant['exp_id'] = exp_id
    variant['tensorboard'] = tensorboard_path

    env_settings = get_env_settings(**env_params)
    variant['Environment'] = env_settings['name']
    algorithm_launcher = launch_settings['algorithm_launcher']


    if icm:
        algorithm_launcher = icm_launcher

    run_experiment(
        algorithm_launcher,
        exp_prefix,
        seed,
        variant,
        **kwargs)


def sweep(exp_prefix, env_params, launch_settings_, tensorboard_path, icm, **kwargs):
    launch_settings = copy.deepcopy(launch_settings_)
    sweeper = launch_settings['sweeper']
    bn_sweeper = launch_settings['batch_norm_sweeper']
    default_params = launch_settings['algo_params']
    default_bn_params = launch_settings['batch_norm_params']
    exp_id = 0
    # So far, only support bn sweeper in random mode
    assert bn_sweeper is None or (
        not isinstance(sweeper, hp.DeterministicHyperparameterSweeper) and
        not isinstance(bn_sweeper, hp.DeterministicHyperparameterSweeper)
    )
    if isinstance(sweeper, hp.DeterministicHyperparameterSweeper):
        print ("yes!")
        for i, params_dict in enumerate(sweeper.iterate_hyperparameters()):
            exp_id += 1
            algo_params = dict(default_params, **params_dict)
            for seed in range(NUM_SEEDS_PER_CONFIG):
                launch_settings['algo_params'] = algo_params
                run_algorithm(launch_settings, env_params, exp_prefix, seed, tensorboard_path + str(i), icm,
                              exp_id=exp_id,
                              **kwargs)
    else:
        for i in range(NUM_HYPERPARAMETER_CONFIGS):
            exp_id += 1
            algo_params = dict(default_params,
                               **sweeper.generate_random_hyperparameters())
            if bn_sweeper is None:
                bn_params = default_bn_params
            else:
                bn_params = dict(default_bn_params,
                                 **bn_sweeper.generate_random_hyperparameters())
            for seed in range(NUM_SEEDS_PER_CONFIG):
                launch_settings['algo_params'] = algo_params
                launch_settings['batch_norm_params'] = bn_params
                run_algorithm(launch_settings, env_params, exp_prefix, seed, tensorboard_path, icm,
                              exp_id=exp_id,
                              **kwargs)


def get_env_params_list_from_args(args):
    envs_params_list = []
    if 'gym' in args.env:
        envs_params_list = [
            dict(
                env_id='gym',
                normalize_env=args.normalize,
                gym_name=gym_name,
            )
            for gym_name in args.gym
        ]

    return envs_params_list + [dict(
        env_id=env,
        normalize_env=args.normalize,
        gym_name="",
    ) for env in args.env if env != 'gym']


def main():
    env_choices = ['ant', 'cheetah', 'cart', 'point', 'reacher', 'idp', 'gym']
    algo_choices = ['ddpg', 'naf', 'shane-ddpg', 'random',
                    'rl-vpg', 'rl-trpo', 'tf-trpo', 'rl-ddpg', 'qddpg', 'oat', 'idle']
    mode_choices = ['local', 'local_docker', 'ec2']
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action='store_true',
                        help="Sweep _hyperparameters for my DDPG.")
    parser.add_argument("--tensorboard", type=str, default='/tmp/',
                        help="Path to store tensorboard diagnostics")
    parser.add_argument("--render", action='store_true',
                        help="Render the environment.")
    parser.add_argument("--env",
                        default=['cart'],
                        help="Environment to test. If env is 'gym' then you "
                             "must pass in argument to the '--gym' option.",
                        nargs='+',
                        choices=env_choices)
    parser.add_argument("--gym",
                        nargs='+',
                        help="Gym environment name (e.g. Cartpole-V1) to test. "
                             "Must pass 'gym' to the '--env' option to use "
                             "this.")
    parser.add_argument("--name", default='default',
                        help='Experiment prefix')
    parser.add_argument("--fast", action='store_true',
                        help=('Run a quick experiment. Intended for debugging. '
                              'Overrides sweep settings'))
    parser.add_argument("--nonorm", action='store_true',
                        help="Normalize the environment")
    parser.add_argument("--algo",
                        default=['ddpg'],
                        help='Algorithm to run.',
                        nargs='+',
                        choices=algo_choices)
    parser.add_argument("--icm", action='store_true',
                        help="Whether to train agent with ICM")
    parser.add_argument("--seed", default=0,
                        type=int,
                        help='Seed')
    parser.add_argument("--num_seeds", default=NUM_SEEDS_PER_CONFIG, type=int,
                        help="Run this many seeds, starting with --seed.")
    parser.add_argument("--mode",
                        default='local',
                        help="Mode to run experiment.",
                        choices=mode_choices,
                        )
    parser.add_argument("--notime", action='store_true',
                        help="Disable time prefix to python command.")
    parser.add_argument("--profile", action='store_true',
                        help="Use cProfile to time the python script.")
    parser.add_argument("--profile_file",
                        help="Where to save .prof file output of cProfiler. "
                             "If set, --profile is forced to be true.")
    args = parser.parse_args()
    args.normalize = not args.nonorm
    args.time = not args.notime

    global N_EPOCHS, EPOCH_LENGTH, EVAL_SAMPLES, MIN_POOL_SIZE
    if args.sweep:
        N_EPOCHS = SWEEP_N_EPOCHS
        MIN_POOL_SIZE = SWEEP_MIN_POOL_SIZE
        EPOCH_LENGTH = SWEEP_EPOCH_LENGTH
        EVAL_SAMPLES = SWEEP_EVAL_SAMPLES
    if args.fast:
        N_EPOCHS = FAST_N_EPOCHS
        EPOCH_LENGTH = FAST_EPOCH_LENGTH
        EVAL_SAMPLES = FAST_EVAL_SAMPLES
        MIN_POOL_SIZE = FAST_MIN_POOL_SIZE

    else:
        if args.render:
            print("WARNING: Algorithm will be slow because render is on.")

    kwargs = dict(
        time=not args.notime,
        save_profile=args.profile or args.profile_file is not None,
        mode=args.mode,
    )
    if args.profile_file:
        kwargs['profile_file'] = args.profile_file
    for env_params in get_env_params_list_from_args(args):
        for launcher_settings in get_launch_settings_list_from_args(args):
            if args.sweep:
                sweep(
                    args.name,
                    env_params,
                    launcher_settings,
                    args.tensorboard,
                    args.icm,
                    **kwargs
                )
            else:
                for i in range(args.num_seeds):
                    run_algorithm(
                        launcher_settings,
                        env_params,
                        args.name,
                        args.seed + i,
                        args.tensorboard + "_"+ str(i),
                        args.icm,
                        **kwargs
                    )


if __name__ == "__main__":
    main()
