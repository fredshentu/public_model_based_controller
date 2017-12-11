"""
Exampling of running DDPG on HalfCheetah.
"""

from railrl.policies.nn_policy import FeedForwardPolicy
from railrl.qfunctions.nn_qfunction import FeedForwardCritic
from railrl.algos.ddpg import DDPG

from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment_here


def example(*_):
    env = HalfCheetahEnv()
    es = OUStrategy(env_spec=env.spec)
    qf = FeedForwardCritic(
        name_or_scope="critic",
        env_spec=env.spec,
    )
    policy = FeedForwardPolicy(
        name_or_scope="actor",
        env_spec=env.spec,
    )
    algorithm = DDPG(
        env,
        es,
        policy,
        qf,
        n_epochs=25,
        batch_size=1024,
        replay_pool_size=10000,
    )
    algorithm.train()


if __name__ == "__main__":
    run_experiment_here(
        example,
        exp_prefix="ddpg-half-cheetah",
        seed=2,
    )
