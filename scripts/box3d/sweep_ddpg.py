import os
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv

from railrl.algos.ddpg import DDPG
from railrl.policies.nn_policy import FeedForwardPolicy
from railrl.qfunctions.nn_qfunction import FeedForwardCritic
from rllab.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import get_env_settings
from railrl.core.tf_util import BatchNormConfig

import itertools

import tensorflow as tf

stub(globals())

# Param ranges
seed = 0
policy_lrs = [1e-5, 1e-4, 1e-3]
qf_lrs = [1e-5, 1e-4, 1e-3]
gammas = [0.9, 0.99, 0.995]
taus = [1e-3, 1e-2]

for policy_lr, qf_lr, gamma, tau in itertools.product(policy_lrs, qf_lrs, gammas, taus):
	env = TfEnv(normalize(env=GymEnv('Box3dReach-v4',record_video=False, \
	log_dir='/tmp/gym_test',record_log=False)))
	
	es = OUStrategy(env_spec=env.spec)
	qf = FeedForwardCritic(
		name_or_scope="critic",
		env_spec=env.spec,
		hidden_nonlinearity=tf.nn.tanh,
	)
	policy = FeedForwardPolicy(
		name_or_scope="actor",
		env_spec=env.spec,
		hidden_nonlinearity=tf.nn.tanh,
	)

	algo = DDPG(
		env,
		es,
		policy,
		qf,
		"/data0/dianchen/box3d/ddpg_box3d_state_v4_tf_policy_{0}_qf_{1}_gamma_{2}_tau_{3}".format(
			policy_lr,
			qf_lr,
			gamma,
			tau,
		),
		qf_learning_rate=qf_lr,
		policy_learning_rate=policy_lr,
		discount=gamma,
		soft_target_tau=tau,
	)

	run_experiment_lite(
		algo.train(),
		exp_prefix="ddpg_box3d_state_v4_tf_policy_{0}_qf_{1}_gamma_{2}_tau_{3}".format(
			policy_lr,
			qf_lr,
			gamma,
			tau,
		),
		n_parallel=1,
		snapshot_mode="last",
		seed=seed,
		mode="local"
	)
