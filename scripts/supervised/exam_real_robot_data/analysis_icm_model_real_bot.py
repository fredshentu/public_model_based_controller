"""
Since the size of real robot data is huge, we first go though all data then save loss array,
then sort loss array. Finally we use the indexes to find the corresponding graphs
"""
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
import argparse
import matplotlib.pyplot as plt
import pickle

OBS_INPUT_SHAPE = [128,128,6]
ACTION_SHAPE = [4]
STATE_SHAPE = [8]

#return img, action, next_img, state, next_state
def load_data(filename):
	obs = []
	next_obs = []
	action = []
	file = open(filename,'rb')
	load_dict = pickle.load(file,encoding='latin1') 
	states = load_dict["states"]
	images = load_dict["images"]
	actions = load_dict["action_list"]
	assert(len(images) == 601)
	assert(len(actions) == 600)
	assert(len(states) == 601)
	return images[:600], actions,images[1:], states[:600], states[1:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the data file, should be pickle')
    
    with tf.Session() as sess:
        data = joblib.load(args.file)
        _conv_encoder = data["encoder"]
        _inverse_model = data["inverse_model"]
        _forward_model = data["forward_model"]
        _state_encoder = data["state_encoder"]
        
    s1_ph = tf.placeholder(tf.float32, [None] + OBS_INPUT_SHAPE)/255 - 0.5
	s2_ph = tf.placeholder(tf.float32, [None] + OBS_INPUT_SHAPE)/255 - 0.5
	a_ph = tf.placeholder(tf.float32, [None, 4]) * [1./1023, 1./249, 1./249, 1./1023]
	arm_state1_ph = tf.placeholder(tf.float32, [None, 8]) / 2048
	arm_state2_ph = tf.placeholder(tf.float32, [None, 8]) / 2048
	
	encoder1 = _conv_encoder.get_weight_tied_copy(observation_input=s1_ph)
	encoder2 = _conv_encoder.get_weight_tied_copy(observation_input=s2_ph)
    state_encoder1 = _state_encoder.get_weight_tied_copy(observation_input=arm_state1_ph)
    state_encoder2 = _state_encoder.get_weight_tied_copy(observation_input=arm_state2_ph)
    feature1 = tf.concat(1, [encoder1.output, state_encoder1.output])
    feature2 = tf.concat(1, [encoder2.output, state_encoder2.output])
    inverse_model = _inverse_model.get_weight_tied_copy(feature_input1=feature1, 
														feature_input2=feature2)
	forward_model = _forward_model.get_weight_tied_copy(feature_input=feature1,
														action_input=a_ph)
														
	def get_forward_loss(obs, state, next_obs, next_state, actions):
		forward_loss = sess.run(
			tf.reduce_mean(tf.square(
				encoder2.output - forward_model.output
			), axis=1),
			feed_dict={
				s1_ph: obs,
				s2_ph: next_obs,
				a_ph: actions,
				arm_state1_ph = state,
				arm_state2_ph = next_state,
			}
		)
		return forward_loss
    
    
    
	# Call rllab rollout for parallel
	while True:
		plt.clf()
		plt.ion()
		ob = env.reset()
		next_ob = None
		x = []
		y = []
		for t in range(env.wrapped_env._wrapped_env.env.spec.max_episode_steps):
			action, _ = policy.get_action(ob)
			next_ob, reward, done, env_infos = env.step(action)
			env.render()
			forward_loss = get_forward_loss([ob], [next_ob], [action])
			if done:
				ob = env.reset()
			else:
				ob = next_ob
			x.append(t)
			y.append(forward_loss)
			# import pdb; pdb.set_trace()
			flag = env_infos["contact_reward"]
			if flag == 1:
				plt.title("touching table")
			if flag == 0:
				plt.title("touching nothing")
			else:
				plt.title("touching box")
			plt.plot(x, y, c="blue")
			plt.pause(0.05)
			# print ("Should plot")
			plt.show()