import numpy as np


class SimpleReplayPool(object):
    def __init__(
            self, max_pool_size, observation_dim, action_dim,
            replacement_policy='stochastic', replacement_prob=1.0,
            max_skip_episode=10, obs_dtype='float32'):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_pool_size = max_pool_size
        self._replacement_policy = replacement_policy
        self._replacement_prob = replacement_prob
        self._max_skip_episode = max_skip_episode
        if hasattr(observation_dim, '__len__'):
            self._observations = np.zeros(
                [max_pool_size] + list(observation_dim), dtype=obs_dtype
            )
        else:
            self._observations = np.zeros(
                [max_pool_size, observation_dim], dtype=obs_dtype
            )
        self._actions = np.zeros(
            [max_pool_size] +  list(action_dim)
        )
        self._rewards = np.zeros(max_pool_size)
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros(max_pool_size, dtype='uint8')
        # self._final_state[i] = state i was the final state in a rollout,
        # so it should never be sampled since it has no correspond
        # In other words, we're saving the s_{t+1} after sampling a tuple of
        # (s_t, a_t, r_t, s_{t+1}, TERMINAL=TRUE)
        self._final_state = np.zeros(max_pool_size, dtype='uint8')
        self._bottom = 0
        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, terminal,
                   final_state):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._final_state[self._top] = final_state
        self.advance()

    def advance(self):
        self._top = (self._top + 1) % self._max_pool_size
        if self._size >= self._max_pool_size:
            self._bottom = (self._bottom + 1) % self._max_pool_size
        else:
            self._size += 1

    def random_batch(self, batch_size):
        assert self._size > 1
        indices = np.zeros(batch_size, dtype='uint64')
        transition_indices = np.zeros(batch_size, dtype='uint64')
        count = 0
        while count < batch_size:
            index = np.random.randint(0, min(self._size, self._max_pool_size))
            # make sure that the transition is valid: if we are at the end of
            # the pool, we need to discard this sample
            if (index + 1) % self._max_pool_size == self._top:
                continue
            # discard the transition if it crosses horizon-triggered resets
            if self._final_state[index]:
                continue
            indices[count] = index
            transition_index = (index + 1) % self._max_pool_size
            transition_indices[count] = transition_index
            count += 1
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._observations[transition_indices]
        )

    @property
    def size(self):
        return self._size


class PixelSimpleReplayPool(SimpleReplayPool):
    def __init__(self, *args, **kwargs):
        super(SimpleReplayPool, self).__init__(*args, **kwargs)
        self._observations = np.zeros(())

class ShaneSimpleReplayPool(object):
    def __init__(
            self, max_pool_size, observation_dim, action_dim,
            replacement_policy='stochastic', replacement_prob=1.0,
            max_skip_episode=10):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_pool_size = max_pool_size
        self._replacement_policy = replacement_policy
        self._replacement_prob = replacement_prob
        self._max_skip_episode = max_skip_episode
        self._observations = np.zeros(
            (max_pool_size, observation_dim),
        )
        self._actions = np.zeros(
            (max_pool_size, action_dim),
        )
        self._rewards = np.zeros(max_pool_size)
        self._terminals = np.zeros(max_pool_size, dtype='uint8')
        self._initials = np.zeros(max_pool_size, dtype='uint8')
        self._bottom = 0
        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, terminal, initial):
        self.check_replacement()
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._initials[self._top] = initial
        self.advance()

    def check_replacement(self):
        if self._replacement_prob < 1.0:
            if self._size < self._max_pool_size or \
                    not self._initials[self._top]: return
            self.advance_until_terminate()

    def get_skip_flag(self):
        if self._replacement_policy == 'full': skip = False
        elif self._replacement_policy == 'stochastic':
            skip = np.random.uniform() > self._replacement_prob
        else: raise NotImplementedError
        return skip

    def advance_until_terminate(self):
        skip = self.get_skip_flag()
        n_skips = 0
        old_top = self._top
        new_top = (old_top + 1) % self._max_pool_size
        while skip and old_top != new_top and n_skips < self._max_skip_episode:
            n_skips += 1
            self.advance()
            while not self._initials[self._top]:
                self.advance()
            skip = self.get_skip_flag()
            new_top = self._top
        logger.log("add_sample, skipped %d episodes, top=%d->%d"%(
            n_skips, old_top, new_top))

    def advance(self):
        self._top = (self._top + 1) % self._max_pool_size
        if self._size >= self._max_pool_size:
            self._bottom = (self._bottom + 1) % self._max_pool_size
        else:
            self._size += 1

    def random_batch(self, batch_size):
        assert self._size > batch_size
        indices = np.zeros(batch_size, dtype='uint64')
        transition_indices = np.zeros(batch_size, dtype='uint64')
        count = 0
        while count < batch_size:
            index = np.random.randint(self._bottom, self._bottom + self._size) % self._max_pool_size
            # make sure that the transition is valid: if we are at the end of the pool, we need to discard
            # this sample
            if index == self._size - 1 and self._size <= self._max_pool_size:
                continue
            # if self._terminals[index]:
            #     continue
            transition_index = (index + 1) % self._max_pool_size
            # make sure that the transition is valid: discard the transition if it crosses horizon-triggered resets
            if not self._terminals[index] and self._initials[transition_index]:
                continue
            indices[count] = index
            transition_indices[count] = transition_index
            count += 1
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            initials=self._initials[indices],
            next_observations=self._observations[transition_indices]
        )

    @property
    def size(self):
        return self._size


class TRPOReplayPool(object):
    """Replay pool"""

    def __init__(
            self, max_pool_size, observation_shape, action_dim,
            obs_dtype,
        ):  # @UndefinedVariable
        self._observation_shape = observation_shape
        self._action_dim = action_dim
        self._max_pool_size = max_pool_size

        self._observations = np.zeros(
            (max_pool_size, observation_shape), dtype=obs_dtype
        )
        self._actions = np.zeros(
            (max_pool_size, action_dim)
        )
        self._rewards = np.zeros(max_pool_size, dtype='float32')
        self._terminals = np.zeros(max_pool_size, dtype='uint8')
        self._bottom = 0
        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, terminal):
        # print (self.observations)
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._top = (self._top + 1) % self._max_pool_size
        if self._size >= self._max_pool_size:
            self._bottom = (self._bottom + 1) % self._max_pool_size
        else:
            self._size = self._size + 1

    def random_batch(self, batch_size):
        assert self._size > batch_size
        indices = np.zeros(batch_size, dtype='uint64')
        transition_indices = np.zeros(batch_size, dtype='uint64')
        count = 0
        while count < batch_size:
            index = np.random.randint(
                self._bottom, self._bottom + self._size) % self._max_pool_size
            # make sure that the transition is valid: if we are at the end of the pool, we need to discard
            # this sample
            if index == self._size - 1 and self._size <= self._max_pool_size:
                continue
            transition_index = (index + 1) % self._max_pool_size
            indices[count] = index
            transition_indices[count] = transition_index
            count += 1
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._observations[transition_indices]
        )

    def mean_obs_act(self):
        if self._size >= self._max_pool_size:
            obs = self._observations
            act = self._actions
        else:
            obs = self._observations[:self._top + 1]
            act = self._actions[:self._top + 1]
        obs_mean = np.mean(obs, axis=0)
        obs_std = np.std(obs, axis=0)
        act_mean = np.mean(act, axis=0)
        act_std = np.std(act, axis=0)
        return obs_mean, obs_std, act_mean, act_std

    @property
    def size(self):
        return self._size