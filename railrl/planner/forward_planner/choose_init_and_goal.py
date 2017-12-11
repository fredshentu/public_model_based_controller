import argparse
import joblib
import uuid
import tensorflow as tf
import pickle
import time
import numpy as np
import gym

PKL_PATH='/home/fred/Desktop/forward_planner_data/std_test_set/'

def env_get_state(env):
	qpos = env.env.model.data.qpos.flat.copy()
	qvel = env.env.model.data.qvel.flat.copy()
	return np.concatenate([qpos, qvel])

def env_set_state(env, ob):
	qpos_idx = env.env.init_qpos.shape[0]
	env.env.set_state(ob[:qpos_idx], ob[qpos_idx:])
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	#Push-v0 and Push-v1 compatiable, even for v1, state should get form v0
	parser.add_argument('env_name', type=str,
						help='name of the env want to use')
	parser.add_argument('pkl_name', type=str,
						help='name of the pickle file')
	parser.add_argument('--hard', action='store_true', help='only use those traj policy takes long time to finish')			
	parser.add_argument('--manually_choose', action='store_true')
	parser.add_argument('--policy', type=str, help='add policy, start applying random action after excuting policy')
	parser.add_argument('--num_tests', type=int, default=10)
	
	args = parser.parse_args()
	env = gym.make(args.env_name)
	
	#choose 20 S_init, S_goal pair, State includes pixel and joint states
	S_init_state_list = []
	S_init_pixel_list = []
	S_goal_state_list = []
	S_goal_pixel_list = []
	real_steps_list = []
	
	S_init_state = None
	S_init_pixel = None
	S_goal_state = None
	S_goal_pixel = None
	
	policy = None
	with tf.Session() as sess:
		if args.policy is not None:
			policy_data = joblib.load(args.policy)
			policy = policy_data['policy']
		if args.manually_choose:
			while True:
				if len(S_init_pixel_list) == args.num_tests:
					break
				obs = env.reset()
				env.render()
				while True:
					nb = input("1: label as S_init, 2: label as S_goal, 3: save S_init S_goal in buffer 4: run policy\n")
					try:
						number = int(nb)
					except ValueError:
						print("you must enter a number")
						break
					if number == 1:
						print("the current state has been saved as s_init")
						S_init_state = env_get_state(env)
						S_init_pixel = env._render('rgb_array')
						
					elif number == 2:
						print("the current state has been saved as s_goal")
						S_goal_state = env_get_state(env)
						S_goal_pixel = env._render('rgb_array')
					elif number == 3:
						print("push this pair to list, index{}".format(len(S_init_state_list)))
						S_init_pixel_list.append(S_init_pixel)
						S_init_state_list.append(S_init_state)
						S_goal_pixel_list.append(S_goal_pixel)
						S_goal_state_list.append(S_goal_state)
						break
					elif number == 4:
						print("policy run 15 times")
						for _ in range(3):
							action = policy.get_action(obs)[0]
							action = np.clip(action, -1, 1)
							obs, r, d, _ = env.step(action)
							env.render()
					elif number == 5:
						#move box upper
						obs[4] += 0.05
						env_set_state(env,obs)
						
					elif number == 6:
						#move box down
						obs[4] -= 0.05
						env_set_state(env, obs)
						
					elif number == 7:
						#move box left
						obs[5] += 0.05
						env_set_state(env, obs)
						
					elif number == 8:
						#move box right
						obs[5] -= 0.05
						env_set_state(env, obs)
						
					elif number == 9:
						print("restart, discard current state")
						break
					elif number == 0:
						for _ in range(3):
							env.step(np.zeros(4))
							env.render()
					else:
						print("random action, next_frame")
						for _ in range(5):
							obs, r, d, _ = env.step(env.action_space.sample())
					env.render()
		else:
			while len(S_init_state_list) < args.num_tests:
				obs = env.reset()
				S_init = obs
				S_init_pixel = env._render('rgb_array')
				env.render()
				#let policy run for 50 times
				print("roll out a trajectory using policy....")
				real_steps = 50
				flag = 1
				for i in range(50):
					action = policy.get_action(obs)[0]
					obs, r, d, _ = env.step(action)
					# print(r)
					#real steps policy needs to take to reach the goal, not accutate....
					if r > -0.05 and flag:
						real_steps = i+1
						break
					# time.sleep(0.1)
					env.render()
				if (np.linalg.norm(obs[4:6] - S_init[4:6]) < 0.0001 and r > -0.05):
					if not args.hard:
						print("num{}".format(len(S_init_state_list) + 1))
						S_init_state_list.append(S_init)
						S_init_pixel_list.append(S_init_pixel)
						S_goal_state_list.append(obs)
						S_goal_pixel_list.append(env._render('rgb_array'))
						print(real_steps)
						real_steps_list.append(real_steps)
					else:
						if (real_steps > 20):
							print("num{}".format(len(S_init_state_list) + 1))
							S_init_state_list.append(S_init)
							S_init_pixel_list.append(S_init_pixel)
							S_goal_state_list.append(obs)
							S_goal_pixel_list.append(env._render('rgb_array'))
							print(real_steps)
							real_steps_list.append(real_steps)
						
	with open(PKL_PATH + args.pkl_name +'.pkl', 'wb') as handle:
		assert(len(S_init_state_list) == len(S_init_pixel_list) and len(S_init_pixel_list) == len(S_goal_state_list) and\
				len(S_goal_state_list) == len(S_goal_pixel_list))
		save_dict = {'S_init_state':S_init_state_list, 'S_init_pixel':S_init_pixel_list,\
						'S_goal_state':S_goal_state_list, 'S_goal_pixel':S_goal_pixel_list, \
						'real_steps_list':real_steps_list}
		pickle.dump(save_dict, handle, pickle.HIGHEST_PROTOCOL)
			
			
			