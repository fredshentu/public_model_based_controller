import joblib
import gym
import numpy as np




def get_state(env):
	d = env.env.model.data
	qpos = d.qpos.flatten()
	qvel = d.qvel.flatten()

	return np.concatenate([qpos, qvel])

def get_ef_pos(env):
    d = env.env.model.data
    xpos = d.site_xpos.flatten()
    return xpos

def get_pos(env):
	d = env.env.model.data
	pos = d.xpos.flatten()
	return pos



env = gym.make('Box3dReach-v12')

states = np.zeros((10, 86))
next_states = np.zeros((10, 86))
actions = np.zeros((10, 4))

for t in range(10):
	env.reset()
	state = get_state(env)
	states[t] = state
	action = env.action_space.sample()
	old_xpos = get_ef_pos(env)

	for _ in range(20):
		env.step(action)
	next_state = get_state(env)
	real_xpos = get_ef_pos(env)

	actions[t] = action
	next_states[t] = next_state

	# env.env.set_state(state[:46], state[46:])

	# print ("Next state:                         ", real_xpos)

	for _ in range(10):
		env.step(env.action_space.sample())
	# print ("Random state:                       ", get_ef_pos(env))
	random_xpos = get_ef_pos(env)
	env.env.set_state(next_state[:46], next_state[46:])
	# for _ in range(1000):
	# 	env.step(np.zeros_like(action))
	# 	env.render()
		# env.env.model.forward()
		# env.env.model.forward()
	# print ("Actuator activation (after 100 zero)", env.env.model.data.act.flatten()[:4])
	# print ("Acceleration (after 1000 zero)", env.env.model.data.qacc.flatten()[:4])
	pred_xpos = get_ef_pos(env)
	# print ("Suppposed to be close to next state:", pred_xpos)
	print ("Error (L2): ", np.linalg.norm(real_xpos - get_ef_pos(env)))
	print ("Error (L2) rel: ", np.linalg.norm(random_xpos - pred_xpos))

	# print ("Qpos Error (L2): ", np.linalg.norm(next_state[:46] - get_state(env)[:46]))
	# print ("Qpos Relative (L2): ", np.linalg.norm(next_state[:46] - state[:46]))
	# # print ("Qpos Error (L2): ", np.linalg.norm(pred_xpos - real_xpos))

# joblib.dump(dict(
# 	states=states,
# 	next_states=next_states,
# 	actions=actions,
# ), '/home/dianchen/rail-rl/scripts/eval_inverse_model/data/test.pkl', compress=3)


# data_dict = joblib.load('/home/dianchen/rail-rl/scripts/eval_inverse_model/data/test.pkl')

# states = data_dict["states"]
# next_states = data_dict['next_states']
# # obs = data_dict['obs']
# # next_obs = data_dict['next_obs']
# actions = data_dict['actions']

# print (states.shape)

# states = states.reshape(states.shape[0]*states.shape[1], states.shape[2])
# next_states = next_states.reshape(next_states.shape[0]*next_states.shape[1], next_states.shape[2])
# obs = obs.reshape([obs.shape[0]*obs.shape[1]] + list(obs.shape[2:]))
# next_obs = next_obs.reshape([next_obs.shape[0]*next_obs.shape[1]] + list(next_obs.shape[2:]))
# actions = actions.reshape(actions.shape[0]*actions.shape[1], actions.shape[2])

# env = gym.make('Box3dReach-v12')
# for state, next_state, action in zip(states, next_states, actions):
# 	env.reset()
# 	env.env.set_state(state[:46], state[46:])
# 	# xpos = get_ef_pos(env)
# 	# state = get_state(env)
# 	old_pos = get_pos(env)
# 	env.step(action)
# 	pred_state = get_state(env)
# 	pred_xpos = get_ef_pos(env)
# 	pred_pos = get_pos(env)
# 	env.env.set_state(next_state[:46], next_state[46:])
# 	print ("Ef pos: ", np.linalg.norm(pred_xpos - get_ef_pos(env)))
# 	print ("qpos: ", np.linalg.norm(pred_state[:46] - get_state(env)[:46]))
# 	print ("qvel: ", np.linalg.norm(pred_state[46:] - get_state(env)[46:]))
# 	print ("xpos: ", np.linalg.norm(pred_pos - get_pos(env)))

# 	print ("prev and next xpos: ", np.linalg.norm(old_pos - get_pos(env)))
