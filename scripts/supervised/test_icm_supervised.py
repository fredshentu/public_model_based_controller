
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize

import argparse
import joblib

import tensorflow as tf
import numpy as np

LOG_FREQ = 1000

def cos_loss(A, B):
	dotproduct = tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(A, 1), tf.nn.l2_normalize(B,1)), axis = 1)
	return 1 - tf.reduce_mean(dotproduct)

def main():

	import matplotlib.pyplot as plt
	plt.ion()

	parser = argparse.ArgumentParser()
	parser.add_argument('env_name', type=str,
						help="name of gym env")
	parser.add_argument('model_path', type=str,
						help="path of trained model")
	parser.add_argument('--cos_forward', action='store_true')
	parser.add_argument('--norm_input', action='store_true')
	parser.add_argument('--mode', type=str, choices=['render', 'record'], default='render')
	parser.add_argument('--data_path', type=str, default='/tmp/data')
	parser.add_argument('--num_sample', type=int, default=100000)

	args = parser.parse_args()

	with tf.Session() as sess:
		data = joblib.load(args.model_path)
		_encoder = data["encoder"]
		_inverse_model = data["inverse_model"]
		_forward_model = data["forward_model"]

		env = TfEnv(normalize(env=GymEnv('Box3dReachPixel-v11',record_video=False, 
				log_dir='/tmp/gym_test',record_log=False)))

		s1_ph = tf.placeholder(tf.float32, [None] + list(env.observation_space.shape))
		s2_ph = tf.placeholder(tf.float32, [None] + list(env.observation_space.shape))

		action_ph = tf.placeholder(tf.float32, [None] + list(env.action_space.shape))

		encoder1 = _encoder.get_weight_tied_copy(observation_input=s1_ph)
		encoder2 = _encoder.get_weight_tied_copy(observation_input=s2_ph)

		inverse_model = _inverse_model.get_weight_tied_copy(feature_input1=encoder1.output,
													  		feature_input2=encoder2.output)
		forward_model = _forward_model.get_weight_tied_copy(feature_input=encoder1.output,
															action_input=action_ph)
		if args.cos_forward:
			forward_loss = cos_loss(encoder2.output, forward_model.output)
		else:
			forward_loss = tf.reduce_mean(tf.square(encoder2.output - forward_model.output))

		inverse_loss = tf.reduce_mean(tf.square(action_ph - inverse_model.output))

		# Start running the env
		obs = env.reset()
		next_obs = None
		x = []
		inverse_losses_results = []
		forward_losses_results = []

		if args.mode == 'render':
			f, (ax1, ax2) = plt.subplots(2)
			ax1.set_title("Inverse loss")
			ax2.set_title("Forward loss")
		elif args.mode == 'record':
			images = np.zeros([args.num_sample, 500, 500, 3], dtype='uint8')
			inverse_losses = np.zeros(args.num_sample, dtype='float32')
			forward_losses = np.zeros(args.num_sample, dtype='float32')
			boxes_contacts = np.zeros(args.num_sample, dtype='uint8')
			table_contacts = np.zeros(args.num_sample, dtype='uint8')

		for t in range(args.num_sample):
			if t % LOG_FREQ == 0:
				print ("Sample: {}".format(t))
			action = env.action_space.sample()
			next_obs, reward, done, env_info = env.step(action)
			if args.mode == 'render':
				env.render()
			elif args.mode == 'record':
				img = env.wrapped_env._wrapped_env.env.env.render(mode='rgb_array')
				images[t, :, :, :] = img

			inverse_loss_result, forward_loss_result = sess.run(
				[inverse_loss, forward_loss],
				{
					s1_ph : [obs / 255.0 - 0.5],
					s2_ph : [next_obs / 255.0 - 0.5],
					action_ph : [action]
				}
			)

			if args.mode == 'render':
				x.append(t)
				inverse_losses_results.append(inverse_loss_result)
				forward_losses_results.append(forward_loss_result)
				ax1.plot(x, inverse_losses_results, c="blue")
				ax2.plot(x, forward_losses_results, c="blue")
				plt.pause(0.001)
				plt.show()
			elif args.mode == 'record':
				boxes_contacts[t] = env_info["contact_reward"]
				table_contacts[t] = env_info["table_contact_reward"]
				forward_losses[t] = forward_loss_result
				inverse_losses[t] = inverse_loss_result
			if done:
				obs = env.reset()
			else:
				obs = next_obs

		if args.mode == 'record':
			data_dict = dict(
				images=images,
				forward_losses=forward_losses,
				inverse_losses=inverse_losses,
				boxes_contacts=boxes_contacts,
				table_contacts=table_contacts
			)
			joblib.dump(data_dict, args.data_path)
			print ("Saved data to {}".format(args.data_path))


if __name__ == "__main__":
	main()