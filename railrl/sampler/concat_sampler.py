import numpy as np
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler

from rllab.sampler.stateful_pool import singleton_pool
from rllab.misc import tensor_utils
from rllab.misc.overrides import overrides



def sample_paths(
        policy_params,
        max_samples,
        max_path_length=np.inf,
        env_params=None,
        scope=None):
    """
    :param policy_params: parameters for the policy. This will be updated on each worker process
    :param max_samples: desired maximum number of samples to be collected. The actual number of collected samples
    might be greater since all trajectories will be rolled out either until termination or until max_path_length is
    reached
    :param max_path_length: horizon / maximum length of a single trajectory
    :return: a list of collected paths
    """
    singleton_pool.run_each(
        _worker_set_policy_params,
        [(policy_params, scope)] * singleton_pool.n_parallel
    )
    if env_params is not None:
        singleton_pool.run_each(
            _worker_set_env_params,
            [(env_params, scope)] * singleton_pool.n_parallel
        )
    return singleton_pool.run_collect(
        _worker_collect_one_path,
        threshold=max_samples,
        args=(max_path_length, scope),
        show_prog_bar=True
    )

def _worker_collect_one_path(G, max_path_length, scope=None):
    G = _get_scoped_G(G, scope)
    path = rollout(G.env, G.policy, max_path_length)
    return path, len(path["rewards"])

def _worker_set_policy_params(G, params, scope=None):
    G = _get_scoped_G(G, scope)
    G.policy.set_param_values(params)

def _get_scoped_G(G, scope):
    if scope is None:
        return G
    if not hasattr(G, "scopes"):
        G.scopes = dict()
    if scope not in G.scopes:
        G.scopes[scope] = SharedGlobal()
        G.scopes[scope].worker_id = G.worker_id
    return G.scopes[scope]

def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1,
            always_return_paths=False):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    o, next_o, d, env_info = env.step(env.action_space.sample())
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o, env_info)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated and not always_return_paths:
        return

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )



class ConcatBatchSampler(BatchSampler):
    def __init__(self, algo):
        super(ConcatBatchSampler, self).__init__(algo)

    @overrides
    def obtain_samples(self, itr):
        cur_params = self.algo.policy.get_param_values()
        paths = sample_paths(
            policy_params=cur_params,
            max_samples=self.algo.batch_size,
            max_path_length=self.algo.max_path_length,
            scope=self.algo.scope,
        )
        if self.algo.whole_paths:
            return paths
        else:
            paths_truncated = parallel_sampler.truncate_paths(paths, self.algo.batch_size)
            return paths_truncated