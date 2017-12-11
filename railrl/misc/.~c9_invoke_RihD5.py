import time
from rllab.core.serializable import Serializable
from numpy.linalg import norm
from numpy import mean
from numpy import std
import numpy as np
import csv, os
import scipy.misc as scm
# import pickle
import tensorflow as tf
from rllab.policies.uniform_control_policy import UniformControlPolicy
from rllab.sampler.utils import rollout
from railrl.policies.cmaes_icm import CMAESPolicy

import matplotlib.pyplot as plt


def save_data(file_name, ob, a, next_ob):
	fieldnames = ["observation", "action", "next_observation"]
	if not os.path.exists(file_name):
		with open(file_name, "w+") as f:
			writer = csv.DictWriter(f, fieldnames)
			writer.writeheader()
	else:
		with open(file_name, "a+") as f:
			writer = csv.DictWriter(f, fieldnames)
			writer.writerow({
				"observation": ob,
				"action": a,
				"next_observation": next_ob,
			})


def get_max_reward(env, policy, num_trajs=200):
	best_reward = 0.0
	for _ in range(num_trajs):
		info = rollout(env, policy)
		print ("Finished traj", _)
		best_reward = max(best_reward, np.sum(info['rewards']))
	print ("Max reward: ", best_reward)

def plot_action(env, policy, save_path, num_actions=1000):
	actions = np.zeros([num_actions, env.spec.action_space.flat_dim])
	o = env.reset()
	for i in range(num_actions):
		a, _ = policy.get_action(o)
		next_o, r, d, env_info = env.step(a)
		actions[i, :] = a
		if d:
			o = env.reset()
		else:
			o = next_o
	np.save(save_path, actions)
	print ("Saved action data")


def test_state_hist(env):
	policy = UniformControlPolicy(env.spec)
	_states = []
	o = env.reset()
	try:
		while True:
			_states.append(o)
			a, _ = policy.get_action(o)
			next_o, r, d, env_info = env.step(a)
			if d:
				o = env.reset()
			else:
				o = next_o
	except KeyboardInterrupt:
		states = np.asarray(_states)
		save_path = '/Users/dianchen/state.npy'
		np.save(save_path, states)
		# pickle.dump(states, save_path)
		print ("State samples saved to {}".format(save_path))


def test_icm_cmaes(encoder, inverse_model, forward_model, env, policy, sess):
	policy = CMAESPolicy(env.spec, encoder, inverse_model, forward_model, sess=sess)
	o = env.reset()
	while True:
		a, _ = policy.get_action([o], env=env)
		next_o, r, d, env_info = env.step(a)
		if d:
			o = env.reset()
		else:
			o = next_o
		env.render()
		time.sleep(0.05)


def investigate_forward_loss(
		_encoder,
		_inverse_model,
		_forward_model,
		env,
		policy,
		sess,
		num_trajs=1,
		num_top=10,
		data_path='/tmp/data/',
		animate=False
	):
	# Rebuild models
	act_space = env.action_space
	obs_space = env.observation_space

	s1_ph = tf.placeholder(tf.float32, [None] + list(obs_space.shape))
	s2_ph = tf.placeholder(tf.float32, [None] + list(obs_space.shape))
	a_ph = tf.placeholder(tf.float32, [None, act_space.flat_dim])
	
	encoder1 = _encoder.get_weight_tied_copy(observation_input=s1_ph)
	encoder2 = _encoder.get_weight_tied_copy(observation_input=s2_ph)
	inverse_model = _inverse_model.get_weight_tied_copy(feature_input1=encoder1.output, 
														feature_input2=encoder2.output)
	forward_model = _forward_model.get_weight_tied_copy(feature_input=encoder1.output,
														action_input=a_ph)

	def get_forward_loss(obs, next_obs, actions, normed=False):
		forward_loss = sess.run(
			tf.reduce_mean(tf.square(
				encoder2.output - forward_model.output
			), axis=1),
			feed_dict={
				s1_ph: obs,
				s2_ph: next_obs,
				a_ph: actions
			}
		)
		if normed:
			nominator = sess.run(
				tf.reduce_mean(tf.square(
					encoder2.output - encoder1.output
				), axis=1),
				feed_dict={
					s1_ph: obs,
					s2_ph: next_obs,
					a_ph: actions
				}
			)
			return forward_loss / nominator
		else:
			return forward_loss

	# Sample actions
	con_obs = []
	con_next_obs = []
	con_actions = []
	non_con_obs = []
	non_con_next_obs = []
	non_con_actions = []
	for n in range(num_trajs):
		print ('Start trajs %d' %n)
		ob = env.reset()
		next_ob = None
		for t in range(env.wrapped_env._wrapped_env.env.spec.max_episode_steps):
			action, _ = policy.get_action(ob)
			next_ob, reward, done, env_infos = env.step(action)
			if animate:
				env.render()
			if env_infos['contact_reward']:
				con_obs.append(ob)
				con_next_obs.append(next_ob)
				con_actions.append(action)
			else:
				non_con_obs.append(ob)
				non_con_next_obs.append(next_ob)
				non_con_actions.append(action)
			if done:
				ob = env.reset()
			else:
				ob = next_ob

	# print (con_obs)
	contacts_fw_losses = list(get_forward_loss(con_obs, con_next_obs, con_actions))
	non_contacts_fw_losses = list(get_forward_loss(non_con_obs, non_con_next_obs, non_con_actions))
	fw_losses = contacts_fw_losses + non_contacts_fw_losses
	obs = con_obs + non_con_obs
	next_obs = con_next_obs + non_con_next_obs
	np_obs = np.array(obs)
	np_next_obs = np.array(next_obs)
	indexes = np.array(fw_losses).argsort()[-num_top:][::-1]
	top_obs = np_obs[indexes]
	top_next_obs = np_next_obs[indexes]

	for i, ob, next_ob in zip(range(num_top), top_obs, top_next_obs):
		env.reset()
		qpos_idx = env.wrapped_env._wrapped_env.env.env.init_qpos.shape[0]
		env.wrapped_env._wrapped_env.env.env.set_state(ob[:qpos_idx], ob[qpos_idx:])
		img_ob = env.wrapped_env._wrapped_env.env.env.render(mode='rgb_array')
		env.wrapped_env._wrapped_env.env.env.set_state(next_ob[:qpos_idx], next_ob[qpos_idx:])
		img_next_ob = env.wrapped_env._wrapped_env.env.env.render(mode='rgb_array')
		img = np.concatenate([img_ob, img_next_ob], axis=1)
		scm.imsave(data_path+'ob_pair{}.png'.format(i), img)
		print ('Saved data {}'.format(i))


	np.save(data_path+'con_int_rew.npy', contacts_fw_losses)
	np.save(data_path+'noncon_int_rew.npy', non_contacts_fw_losses)
	print ("Saved losses data")

def plot_forward(_encoder, _inverse_model, _forward_model, env, policy, sess):
	from railrl.misc.pyhelper_fns.vis_utils import MyAnimationMulti
	vis_tool = MyAnimationMulti(None, numPlots = 2, isIm = [1,0])
	# Rebuild models
	act_space = env.action_space
	obs_space = env.observation_space

	s1_ph = tf.placeholder(tf.float32, [None] + list(obs_space.shape))
	s2_ph = tf.placeholder(tf.float32, [None] + list(obs_space.shape))
	a_ph = tf.placeholder(tf.float32, [None, act_space.flat_dim])
	
	encoder1 = _encoder.get_weight_tied_copy(observation_input=s1_ph)
	encoder2 = _encoder.get_weight_tied_copy(observation_input=s2_ph)
	inverse_model = _inverse_model.get_weight_tied_copy(feature_input1=encoder1.output, 
														feature_input2=encoder2.output)
	forward_model = _forward_model.get_weight_tied_copy(feature_input=encoder1.output,
														action_input=a_ph)

	def get_forward_loss(obs, next_obs, actions):
		forward_loss = sess.run(
			tf.reduce_mean(tf.square(
				encoder2.output - forward_model.output
			), axis=1),
			feed_dict={
				s1_ph: obs,
				s2_ph: next_obs,
				a_ph: actions
			}
		)
		return forward_loss

	# Call rllab rollout for parallel
	while True:
		ob = env.reset()
		next_ob = None
		x = []
		y = []
		for t in range(env.wrapped_env._wrapped_env.env.spec.max_episode_steps):
			print()
			action, _ = policy.get_action(ob)
			next_ob, reward, done, env_infos = env.step(action)
			# import pdb; pdb.set_trace()
			image = env._wrapped_env._wrapped_env.env.env.render(mode='rgb_array')
			forward_loss = get_forward_loss([ob], [next_ob], [action])
			if done:
				ob = env.reset()
			else:
				ob = next_ob

			x.append(t)
			y.append(forward_loss)
			vis_tool._display([image, [y]])


def get_time_to_first_contact(env, policy, is_random=False, num_trajs=100):
	import itertools
	time_contact = []
	if is_random:
		from rllab.policies.uniform_control_policy import UniformControlPolicy
		policy = UniformControlPolicy(env.spec)
	print ("Using {}".format(policy))
	for traj_i in range(num_trajs):
		obs = env.reset()
		print ("Start traj {}".format(traj_i))
		for t in itertools.count():
			action, _ = policy.get_action(obs)
			obs, reward, done, env_info = env.step(action)
			if env_info['contact_reward'] > 0 or done:
				time_contact.append(t)
				break
	# plt.hist(time_contact)
	# plt.title("Time to first contact over {} trajectories".format(num_trajs))
	# plt.show()
	data_path = input("Where do you want to save it? \n")
	np.save(data_path, time_contact)
	print ("Data saved")
	print ("Mean time to first contact: {}, median:{}, std:{} for {}, ({} trajectories)".format(np.mean(time_contact), np.median(time_contact), np.std(time_contact), policy, num_trajs))


def episode_reward(env, policy, is_random=False):
	import itertools
	mean_reward = []
	if is_random:
		from rllab.policies.uniform_control_policy import UniformControlPolicy
		policy = UniformControlPolicy(env.spec)
	print ("Using {}".format(policy))
	for traj_i in range(num_trajs):
		obs = env.reset()
		print ("Start traj {}".format(traj_i))
		rewards
		for t in itertools.count():
			action, _ = policy.get_action(obs)
			obs, reward, done, env_info = env.step(action)
			if done:
				break
	plt.his
	print ("Mean time to first contact: {} for {}, ({} trajectories)".format(np.mean(time_contact), policy, num_trajs))

def investigate_inverse_loss(
		_encoder, 
		_inverse_model, 
		_forward_model, 
		env, 
		policy, 
		sess, 
		img_path='/tmp/img', 
		num_trajs=100, 
		num_top=10,
		animate=False
	):
	# Rebuild models
	act_space = env.action_space
	obs_space = env.observation_space

	s1_ph = tf.placeholder(tf.float32, [None] + list(obs_space.shape))
	s2_ph = tf.placeholder(tf.float32, [None] + list(obs_space.shape))
	a_ph = tf.placeholder(tf.float32, [None, act_space.flat_dim])
	
	encoder1 = _encoder.get_weight_tied_copy(observation_input=s1_ph)
	encoder2 = _encoder.get_weight_tied_copy(observation_input=s2_ph)
	inverse_model = _inverse_model.get_weight_tied_copy(feature_input1=encoder1.output, 
														feature_input2=encoder2.output)
	forward_model = _forward_model.get_weight_tied_copy(feature_input=encoder1.output,
														action_input=a_ph)
	def get_inverse_loss(obs, next_obs, actions):
		return sess.run(
			tf.reduce_mean(
				tf.square(a_ph - inverse_model.output),
				axis=1
			), 
			feed_dict={
				s1_ph: obs,
				s2_ph: next_obs,
				a_ph: actions
			}
		)

	# Sample trajectories
	obs = []
	next_obs = []
	actions = []
	# Call rllab rollout for parallel
	for n in range(num_trajs):
		print ('Start trajs %d' %n)
		ob = env.reset()
		next_ob = None
		for t in range(env.wrapped_env._wrapped_env.env.spec.max_episode_steps):
			action, _ = policy.get_action(ob)
			next_ob, reward, done, env_infos = env.step(action)
			if animate:
				env.render()
			obs.append(ob)
			next_obs.append(next_ob)
			actions.append(action)
			if done:
				ob = env.reset()
			else:
				ob = next_ob

	inverse_loss = get_inverse_loss(obs, next_obs, actions)

	np_obs = np.array(obs)
	np_next_obs = np.array(next_obs)
	np_actions = np.array(actions)
	indexes = inverse_loss.argsort()[-num_top:][::-1]
	top_obs = np_obs[indexes]
	top_next_obs = np_next_obs[indexes]
	top_actions = np_actions[indexes]

	for i, ob, next_ob, a in zip(range(num_top), top_obs, top_next_obs, top_actions):
		env.reset()
		env.wrapped_env._wrapped_env.env.env.set_state(ob[:28], ob[28:])
		img_ob = env.wrapped_env._wrapped_env.env.env.render(mode='rgb_array')
		env.wrapped_env._wrapped_env.env.env.set_state(next_ob[:28], next_ob[28:])
		img_next_ob = env.wrapped_env._wrapped_env.env.env.render(mode='rgb_array')
		img = np.concatenate([img_ob, img_next_ob], axis=1)
		scm.imsave(img_path+'ob_pair{}.png'.format(i), img)
		np.save(img_path+'torc_{}.npy'.format(i), a)
		print ('Saved data {}'.format(i))



def test_icm(_encoder, _inverse_model, _forward_model, env, policy, sess, animate=False):
	# Rebuild models
	act_space = env.action_space
	obs_space = env.observation_space

	s1_ph = tf.placeholder(tf.float32, [None] + list(obs_space.shape))
	s2_ph = tf.placeholder(tf.float32, [None] + list(obs_space.shape))
	a_ph = tf.placeholder(tf.float32, [None, act_space.flat_dim])
	
	encoder1 = _encoder.get_weight_tied_copy(observation_input=s1_ph)
	encoder2 = _encoder.get_weight_tied_copy(observation_input=s2_ph)
	inverse_model = _inverse_model.get_weight_tied_copy(feature_input1=encoder1.output, 
														feature_input2=encoder2.output)
	forward_model = _forward_model.get_weight_tied_copy(feature_input=encoder1.output,
														action_input=a_ph)

	def apply_action(env, a):
		old_data = env.wrapped_env.env.env.data
		old_qpos = old_data.qpos.flatten()
		old_qvel = old_data.qvel.flatten()
		next_o, r, d, env_info = env.step(a)
		env.wrapped_env.env.env.set_state(old_qpos, old_qvel)
		env.wrapped_env.env._elapsed_steps -= 1
		return next_o

	def get_forward_loss(ob, next_ob, a):
		forward_loss = tf.nn.l2_loss(encoder2.output - forward_model.output)
		return sess.run(forward_loss, {
				encoder1.observation_input: [ob],
				encoder2.observation_input: [next_ob],
				forward_model.action_input: [a],
			})

	def get_pred_f2(f1, a):
		return sess.run(forward_model.output, { forward_model.feature_input: [f1],
												forward_model.action_input: [a] })

	o = env.reset()
	f_difference = [] # Difference of f1 - f2
	pred_difference = [] # Difference of f2_pred - f2
	try:
		print ("Press ctrl+c to exit")
		while True:
			a, _ = policy.get_action(o)
			f1 = _encoder.get_features(o)
			# f2_pred = get_pred_f2(f1, a)
			f2 = _encoder.get_features(apply_action(env, a))
			# f_difference.append(norm(f1 - f2)**2/2)
			pred_difference.append(get_forward_loss(o, apply_action(env, a), a))
			next_o, r, d, env_info = env.step(a)
			if d:
				o = env.reset()
			else:
				o = next_o
			if animate:
				env.render()
				time.sleep(0.05)

	except KeyboardInterrupt:
		print ("\n----------")
		print ("Collected {} samples".format(len(f_difference)))
		print ("Mean of f1 - f2 difference: {}".format(mean(f_difference)))
		print ("Std of f1 - f2 difference: {}".format(std(f_difference)))
		print ("Mean of f2pred - f2 difference: {}".format(mean(pred_difference)))
		print ("Std of f2pred - f2 difference: {}".format(std(pred_difference)))



