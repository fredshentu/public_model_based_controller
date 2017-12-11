"""
This file contains classic RL launchers. See module docstring for more detail.
"""
from rllab.misc import logger
from rllab.misc.instrument import run_experiment_lite


#################
# my algorithms #
#################

def icm_launcher(variant):

	if variant["Algorithm"] == "DDPG":
		from railrl.algos.ddpg import DDPG as MyDDPG
		from railrl.policies.nn_policy import FeedForwardPolicy
		from railrl.qfunctions.nn_qfunction import FeedForwardCritic
		from rllab.exploration_strategies.ou_strategy import OUStrategy
		from railrl.exploration_strategies.simple_gaussian_strategy import SimpleGaussianStrategy
		from railrl.launchers.launcher_util import get_env_settings
		from railrl.core.tf_util import BatchNormConfig
		from railrl.algos.icm import ICM

		if ('batch_norm_params' in variant
			and variant['batch_norm_params'] is not None):
			bn_config = BatchNormConfig(**variant['batch_norm_params'])
		else:
			bn_config = None
		env_settings = get_env_settings(**variant['env_params'])
		env = env_settings['env']
		es = OUStrategy(env_spec=env.spec)
		# es = SimpleGaussianStrategy(env_spec=env.spec, sigma=0.5)
		qf = FeedForwardCritic(
			name_or_scope="critic",
			env_spec=env.spec,
			batch_norm_config=bn_config,
			**variant.get('qf_params', {})
		)
		policy = FeedForwardPolicy(
			name_or_scope="actor",
			env_spec=env.spec,
			batch_norm_config=bn_config,
			**variant.get('policy_params', {})
		)

		algo = MyDDPG(
			env,
			es,
			policy,
			qf,
			variant['tensorboard'],
			batch_norm_config=bn_config,
			**variant['algo_params'],
		)
		algorithm = ICM(
			env, 
			algo,
			no_encoder=False,
			feature_dim=env.spec.observation_space.flat_dim, 
			forward_weight=0.9,
			external_reward_weight=0.95,
			inverse_tanh=True,
			init_learning_rate=1e-3
		)
		algorithm.train()
	elif variant["Algorithm"] == "Idle":
		from railrl.algos.idle import IdleAlgo
		from railrl.launchers.launcher_util import get_env_settings
		from railrl.algos.icm import ICM
		env_settings = get_env_settings(**variant['env_params'])
		env = env_settings['env']
		algo = IdleAlgo(env, variant['tensorboard'])
		algorithm = ICM(
			env, 
			algo,
			no_encoder=False,
			feature_dim=env.spec.observation_space.flat_dim,
			forward_weight=0.9,
			external_reward_weight=0.0,
			inverse_tanh=True,
			init_learning_rate=1e-3,
		)
		algorithm.train()
	elif variant["Algorithm"] == "rllab-TRPO":
		from rllab.algos.trpo import TRPO
		from railrl.launchers.launcher_util import get_env_settings
		from railrl.algos.icm_trpo import ICM
		from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
		from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
		from railrl.algos.icm_trpo import ICM
		import lasagne.nonlinearities as NL

		env_settings = get_env_settings(**variant['env_params'])
		env = env_settings['env']
		policy = GaussianMLPPolicy(
			env_spec=env.spec,
			hidden_sizes=(64, 32),
			output_nonlinearity=NL.tanh,
		)

		baseline = LinearFeatureBaseline(
			env.spec,
		)

		batch_size = 5000
		algo = TRPO(
			env=env,
			policy=policy,
			baseline=baseline,
			batch_size=batch_size,
			whole_paths=True,
			max_path_length=1000,
			n_itr=1000,
			step_size=0.01,
			subsample_factor=1.0,
		)
		algorithm = ICM(
			env, 
			algo,
			variant['tensorboard'],
			no_encoder=False,
			feature_dim=env.spec.observation_space.flat_dim,
			forward_weight=0.2,
			external_reward_weight=0.99,
			inverse_tanh=True,
			init_learning_rate=1e-4,
		)
		algorithm.train()

	elif variant["Algorithm"] == 'tf-TRPO':
		from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
		from sandbox.rocky.tf.baselines.gaussian_conv_baseline import GaussianConvBaseline
		from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
		from sandbox.rocky.tf.policies.gaussian_conv_policy import GaussianConvPolicy
		from sandbox.rocky.tf.algos.trpo import TRPO
		from sandbox.rocky.tf.envs.base import TfEnv

		from railrl.launchers.launcher_util import get_env_settings
		# from railrl.algos.icm_trpo_tf import ICM
		from railrl.algos.icm_trpo_tf_box3d import ICM
		import tensorflow as tf

		env_settings = get_env_settings(**variant['env_params'])
		env = TfEnv(env_settings['env'])
		if len(env.observation_space.shape) == 1:
			policy = GaussianMLPPolicy(
				"mlp_policy",
				env_spec=env.spec,
				hidden_sizes=(64, 32),
				output_nonlinearity=tf.nn.tanh,
			)
			baseline = LinearFeatureBaseline(
				env.spec,
			)
		elif len(env.observation_space.shape) == 2:
			policy = ConvNNPolicy(
				"conv_policy",
				env_spec=mdp.spec,
				conv_filters=(32, 32, 32, 32),
				conv_filter_sizes=((3,3),(3,3),(3,3),(3,3)),
				conv_strides=(2, 2, 2, 2),
				conv_pads=('SAME', 'SAME', 'SAME', 'SAME'),
				hidden_sizes=(256,),
			)

			baseline = GaussianConvBaseline(
				mdp.spec,
				regressor_args={
					'conv_filters':(32, 32, 32, 32),
					'conv_filter_sizes':((3,3),(3,3),(3,3),(3,3)),
					'conv_strides':(2, 2, 2, 2),
					'conv_pads':('SAME', 'SAME', 'SAME', 'SAME'),
					'hidden_sizes':(256,),
				}
			)
		else:
			raise NotImplementedError("Sorry, no support for observatin space: {}".format(env.observation_space.shape))

		batch_size = 5000
		algo = TRPO(
			env=env,
			policy=policy,
			baseline=baseline,
			batch_size=batch_size,
			whole_paths=True,
			max_path_length=500,
			n_itr=1000,
			step_size=0.01,
			subsample_factor=1.0,
		)

		algorithm = ICM(
			env, 
			algo,
			variant['tensorboard'],
			no_encoder=False,
			feature_dim=env.spec.observation_space.flat_dim,
			forward_weight=0.2,
			external_reward_weight=0.99,
			inverse_tanh=True,
			init_learning_rate=1e-4
		)
		algorithm.train()

	else:
		raise NotImplementedError("Currently only supports DDPG!")


def my_ddpg_launcher(variant):
	"""
	Run DDPG
	:param variant: Dictionary of dictionary with the following keys:
		- algo_params
		- env_params
		- qf_params
		- policy_params
	:return:
	"""
	from railrl.algos.ddpg import DDPG as MyDDPG
	from railrl.policies.nn_policy import FeedForwardPolicy
	from railrl.qfunctions.nn_qfunction import FeedForwardCritic
	from rllab.exploration_strategies.ou_strategy import OUStrategy
	from railrl.launchers.launcher_util import get_env_settings
	from railrl.core.tf_util import BatchNormConfig
	if ('batch_norm_params' in variant
		and variant['batch_norm_params'] is not None):
		bn_config = BatchNormConfig(**variant['batch_norm_params'])
	else:
		bn_config = None
	env_settings = get_env_settings(**variant['env_params'])
	env = env_settings['env']
	es = OUStrategy(env_spec=env.spec)
	qf = FeedForwardCritic(
		name_or_scope="critic",
		env_spec=env.spec,
		batch_norm_config=bn_config,
		**variant.get('qf_params', {})
	)
	policy = FeedForwardPolicy(
		name_or_scope="actor",
		env_spec=env.spec,
		batch_norm_config=bn_config,
		**variant.get('policy_params', {})
	)

	algorithm = MyDDPG(
		env,
		es,
		policy,
		qf,
		variant['tensorboard'],
		batch_norm_config=bn_config,
		**variant['algo_params'],
	)
	algorithm.train()


def quadratic_ddpg_launcher(variant):
	"""
	Run DDPG with Quadratic Critic
	:param variant: Dictionary of dictionary with the following keys:
		- algo_params
		- env_params
		- qf_params
		- policy_params
	:return:
	"""
	from railrl.algos.ddpg import DDPG as MyDDPG
	from railrl.policies.nn_policy import FeedForwardPolicy
	from rllab.exploration_strategies.ou_strategy import OUStrategy
	from railrl.qfunctions.quadratic_naf_qfunction import QuadraticNAF
	from railrl.launchers.launcher_util import get_env_settings
	from railrl.core.tf_util import BatchNormConfig
	if ('batch_norm_params' in variant
		and variant['batch_norm_params'] is not None):
		bn_config = BatchNormConfig(**variant['batch_norm_params'])
	else:
		bn_config = None
	env_settings = get_env_settings(**variant['env_params'])
	env = env_settings['env']
	es = OUStrategy(env_spec=env.spec)
	qf = QuadraticNAF(
		name_or_scope="critic",
		env_spec=env.spec,
		batch_norm_config=bn_config,
		**variant['qf_params']
	)
	policy = FeedForwardPolicy(
		name_or_scope="actor",
		env_spec=env.spec,
		batch_norm_config=bn_config,
		**variant['policy_params']
	)
	algorithm = MyDDPG(
		env,
		es,
		policy,
		qf,
		batch_norm_config=bn_config,
		**variant['algo_params']
	)
	algorithm.train()


def oat_qddpg_launcher(variant):
	"""
	Quadratic optimal action target DDPG
	"""
	from railrl.algos.optimal_action_target_ddpg import OptimalActionTargetDDPG as OAT
	from railrl.policies.nn_policy import FeedForwardPolicy
	from railrl.qfunctions.quadratic_naf_qfunction import QuadraticNAF
	from rllab.exploration_strategies.ou_strategy import OUStrategy
	from railrl.launchers.launcher_util import get_env_settings
	from railrl.core.tf_util import BatchNormConfig
	if ('batch_norm_params' in variant
		and variant['batch_norm_params'] is not None):
		bn_config = BatchNormConfig(**variant['batch_norm_params'])
	else:
		bn_config = None
	env_settings = get_env_settings(**variant['env_params'])
	env = env_settings['env']
	es = OUStrategy(env_spec=env.spec)
	qf = QuadraticNAF(
		name_or_scope="critic",
		env_spec=env.spec,
		batch_norm_config=bn_config,
		**variant['qf_params']
	)
	policy = FeedForwardPolicy(
		name_or_scope="actor",
		env_spec=env.spec,
		batch_norm_config=bn_config,
		**variant['policy_params']
	)
	algorithm = OAT(
		env,
		es,
		policy,
		qf,
		batch_norm_config=bn_config,
		**variant['algo_params']
	)
	algorithm.train()


def naf_launcher(variant):
	from railrl.algos.naf import NAF
	from railrl.qfunctions.quadratic_naf_qfunction import QuadraticNAF
	from rllab.exploration_strategies.ou_strategy import OUStrategy
	from railrl.launchers.launcher_util import get_env_settings
	from railrl.core.tf_util import BatchNormConfig
	if ('batch_norm_params' in variant
			and variant['batch_norm_params'] is not None):
		bn_config = BatchNormConfig(**variant['batch_norm_params'])
	else:
		bn_config = None
	env_settings = get_env_settings(**variant['env_params'])
	env = env_settings['env']
	if 'es_init' in variant:
		es = variant['es_init'](env, **variant['exploration_strategy_params'])
	else:
		es = OUStrategy(
			env_spec=env.spec,
			**variant['exploration_strategy_params']
		)
	qf = QuadraticNAF(
		name_or_scope="qf",
		env_spec=env.spec,
		batch_norm_config=bn_config,
	)
	algorithm = NAF(
		env,
		es,
		qf,
		batch_norm_config=bn_config,
		**variant['algo_params']
	)
	algorithm.train()


def get_naf_ddpg_params():
	import tensorflow as tf
	# TODO: try this
	variant = {
		'Algorithm': 'NAF-DDPG',
		'quadratic_policy_params': dict(
			observation_hidden_sizes=(100, 100),
			hidden_W_init=None,
			hidden_b_init=None,
			output_W_init=None,
			output_b_init=None,
			hidden_nonlinearity=tf.nn.relu,
			output_nonlinearity=tf.nn.tanh,
		),
		'policy_params': dict(
			observation_hidden_sizes=(100, 100),
			hidden_nonlinearity=tf.nn.relu,
			output_nonlinearity=tf.nn.tanh,
		)
	}
	return variant

####################
# other algorithms #
####################
def shane_ddpg_launcher(variant):
	from rllab.exploration_strategies.gaussian_strategy import GaussianStrategy
	from sandbox.rocky.tf.algos.ddpg import DDPG as ShaneDDPG
	from sandbox.rocky.tf.envs.base import TfEnv
	from sandbox.rocky.tf.policies.deterministic_mlp_policy import (
		DeterministicMLPPolicy
	)
	from sandbox.rocky.tf.q_functions.continuous_mlp_q_function import (
		ContinuousMLPQFunction
	)
	from railrl.launchers.launcher_util import get_env_settings
	env_settings = get_env_settings(**variant['env_params'])
	env = TfEnv(env_settings['env'])
	es = GaussianStrategy(env.spec)

	policy = DeterministicMLPPolicy(
		name="init_policy",
		env_spec=env.spec,
		**variant['policy_params']
	)
	qf = ContinuousMLPQFunction(
		name="qf",
		env_spec=env.spec,
		**variant['qf_params']
	)

	algorithm = ShaneDDPG(
		env,
		policy,
		qf,
		es,
		**variant['algo_params']
	)
	algorithm.train()


def rllab_vpg_launcher(variant):
	from rllab.algos.trpo import TRPO
	from railrl.launchers.launcher_util import get_env_settings
	from railrl.algos.icm_trpo import ICM
	from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
	from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
	env_settings = get_env_settings(**variant['env_params'])
	env = TfEnv(env_settings['env'])
	policy = GaussianMLPPolicy(
		name="policy",
		env_spec=env.spec,
		hidden_sizes=(32, 32)
	)

	baseline = LinearFeatureBaseline(env_spec=env.spec)

	algorithm = VPG(
		env=env,
		policy=policy,
		baseline=baseline,
		**variant['algo_params']
	)
	algorithm.train()


def rllab_trpo_launcher(variant):
	from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
	from rllab.algos.trpo import TRPO
	from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
	

	from railrl.launchers.launcher_util import get_env_settings
	import lasagne.nonlinearities as NL

	env_settings = get_env_settings(**variant['env_params'])
	env = env_settings['env']
	policy = GaussianMLPPolicy(
		env_spec=env.spec,
		hidden_sizes=(64, 32),
		output_nonlinearity=NL.tanh,
	)
	baseline = LinearFeatureBaseline(
		env.spec,
	)
	batch_size = 5000
	algorithm = TRPO(
		env=env,
		policy=policy,
		baseline=baseline,
		whole_paths=True,
		max_path_length=500,
		n_itr=1000,
		step_size=0.01,
		subsample_factor=1.0,
	)
	algorithm.train()

def tf_trpo_launcher(variant):
	from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
	from sandbox.rocky.tf.baselines.gaussian_conv_baseline import GaussianConvBaseline
	from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
	from sandbox.rocky.tf.policies.gaussian_conv_policy import GaussianConvPolicy
	from sandbox.rocky.tf.algos.trpo import TRPO
	from sandbox.rocky.tf.envs.base import TfEnv

	from railrl.launchers.launcher_util import get_env_settings
	import tensorflow as tf

	env_settings = get_env_settings(**variant['env_params'])
	env = TfEnv(env_settings['env'])
	if len(env.observation_space.shape) == 1:
		policy = GaussianMLPPolicy(
			"mlp_policy",
			env_spec=env.spec,
			hidden_sizes=(64, 32),
			output_nonlinearity=tf.nn.tanh,
		)
		baseline = LinearFeatureBaseline(
			env.spec,
		)
	elif len(env.observation_space.shape) == 2:
		policy = ConvNNPolicy(
			"conv_policy",
			env_spec=mdp.spec,
			conv_filters=(32, 32, 32, 32),
			conv_filter_sizes=((3,3),(3,3),(3,3),(3,3)),
			conv_strides=(2, 2, 2, 2),
			conv_pads=('SAME', 'SAME', 'SAME', 'SAME'),
			hidden_sizes=(256,),
		)

		baseline = GaussianConvBaseline(
			mdp.spec,
			regressor_args={
				'conv_filters':(32, 32, 32, 32),
				'conv_filter_sizes':((3,3),(3,3),(3,3),(3,3)),
				'conv_strides':(2, 2, 2, 2),
				'conv_pads':('SAME', 'SAME', 'SAME', 'SAME'),
				'hidden_sizes':(256,),
			}
		)
	else:
		raise NotImplementedError("Sorry, no support for observatin space: {}".format(env.observation_space.shape))

	batch_size = 5000
	algorithm = TRPO(
		env=env,
		policy=policy,
		baseline=baseline,
		batch_size=batch_size,
		whole_paths=True,
		max_path_length=500,
		n_itr=1000,
		step_size=0.01,
		subsample_factor=1.0,
	)

	algorithm.train()


def rllab_ddpg_launcher(variant):
	from rllab.algos.ddpg import DDPG as RllabDDPG
	from rllab.exploration_strategies.ou_strategy import OUStrategy
	from rllab.q_functions.continuous_mlp_q_function import (
		ContinuousMLPQFunction as TheanoContinuousMLPQFunction
	)
	from rllab.policies.deterministic_mlp_policy import (
		DeterministicMLPPolicy as TheanoDeterministicMLPPolicy
	)
	from railrl.launchers.launcher_util import get_env_settings
	env_settings = get_env_settings(**variant['env_params'])
	env = env_settings['env']
	policy = TheanoDeterministicMLPPolicy(
		env_spec=env.spec,
		hidden_sizes=(32, 32)
	)

	es = OUStrategy(env_spec=env.spec)

	qf = TheanoContinuousMLPQFunction(env_spec=env.spec)

	algorithm = RllabDDPG(
		env=env,
		policy=policy,
		es=es,
		qf=qf,
		**variant['algo_params']
	)
	algorithm.train()


def random_action_launcher(variant):
	from railrl.algos.noop_algo import NoOpAlgo
	from rllab.exploration_strategies.ou_strategy import OUStrategy
	from rllab.policies.uniform_control_policy import UniformControlPolicy
	from railrl.launchers.launcher_util import get_env_settings
	env_settings = get_env_settings(**variant['env_params'])
	env = env_settings['env']
	es = OUStrategy(env)
	policy = UniformControlPolicy(env_spec=env.spec)
	algorithm = NoOpAlgo(
		env,
		policy,
		es,
		**variant['algo_params']
	)
	algorithm.train()


def run_experiment(
		task,
		exp_prefix,
		seed,
		variant,
		time=True,
		save_profile=False,
		profile_file='time_log.prof',
		**kwargs):
	"""

	:param task:
	:param exp_prefix:
	:param seed:
	:param variant:
	:param time: Add a "time" command to the python command?
	:param save_profile: Create a cProfile log?
	:param kwargs:
	:return:
	"""
	variant['seed'] = str(seed)
	logger.log("Variant:")
	logger.log(str(variant))
	command_words = []
	if time:
		command_words.append('time')
	command_words.append('python')
	if save_profile:
		command_words += ['-m cProfile -o', profile_file]
	run_experiment_lite(
		task,
		snapshot_mode="last",
		exp_prefix=exp_prefix,
		variant=variant,
		seed=seed,
		use_cloudpickle=True,
		python_command=' '.join(command_words),
		**kwargs
	)
