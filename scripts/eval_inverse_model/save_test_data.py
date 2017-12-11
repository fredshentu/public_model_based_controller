import joblib
import gym
import numpy as np

def get_state(env):
	d = env.env.model.data
	qpos = d.qpos.flatten()
	qvel = d.qvel.flatten()

	return np.concatenate([qpos, qvel])


env = gym.make('Box3dReach-v12')

states = np.zeros((1000, 86))
next_states = np.zeros((1000, 86))
actions = np.zeros((1000, 4))
obs = np.zeros((1000, 86))
next_obs = np.zeros((1000, 86))

ob = env.reset()
for t in range(1000):
	states[t] = get_state(env)
	obs[t] = ob
	action = env.action_space.sample()
	actions[t] = action
	next_ob, _, _, _ = env.step(action)
	next_obs[t] = next_ob
	next_states[t] = get_state(env)

joblib.dump(dict(
	states=states,
	next_states=next_states,
	actions=actions,
	obs=obs,
	next_obs=next_obs,
), '/home/dianchen/rail-rl/scripts/eval_inverse_model/data/test.pkl', compress=3)
