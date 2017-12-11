import argparse
import joblib
import uuid
import tensorflow as tf
import pickle
import time
import numpy as np
import gym

def env_get_state(env):
	qpos = env.env.model.data.qpos.flat.copy()
	qvel = env.env.model.data.qvel.flat.copy()
	return np.concatenate([qpos, qvel])

def env_set_state(env, ob):
	qpos_idx = env.env.init_qpos.shape[0]
	env.env.set_state(ob[:qpos_idx], ob[qpos_idx:])
	
def env_set_and_render(env, state):
	env_set_state(env, state)
	env.render()
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('env_name', type=str,
						help='name of the env want to use')
	parser.add_argument('save_path', type=str,
						help='where to save')
	parser.add_argument('policy', type=str, help='add policy, start applying random action after excuting policy')
	parser.add_argument('--num_tests', type=int, default=10000)
	
	args = parser.parse_args()
	env = gym.make(args.env_name)

	S_init_list = []
	S_goal_list = []
	

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)

	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		#load policy 
		policy_data = joblib.load(args.policy)
		policy = policy_data['policy']
		
		counter = 0
		while counter < args.num_tests:
			print(counter)
			obs = env.reset()
			real_state = env_get_state(env)
			if np.random.rand() < 0.15:
				S_init_list.append(real_state)
				for _ in range(50):
					env.step(env.action_space.sample())
				S_goal_list.append(env_get_state(env))
				counter += 1
				# print("S_init")
				# env_set_and_render(env, real_state)
				# print("S_goal")
				# env_set_and_render(env, S_goal_list[-1])
				# import pdb; pdb.set_trace()
			else:
				policy_traj_state = [real_state]
				policy_traj_contact = [False]
				policy_contact_counter = 0
				for i in range(80):
					obs, r, d, env_info = env.step(policy.get_action(obs)[0])
					policy_traj_state.append(env_get_state(env))
					if env_info['contact']:
						policy_traj_contact.append(True)
						policy_contact_counter += 1
					else:
						policy_traj_contact.append(False)
				if policy_contact_counter > 7:
				#check the policy trajectory choose init and goal state from policy trajectory
					first_touch_index = policy_traj_contact.index(True)
					S_init_list.append(policy_traj_state[first_touch_index - 1])
					goal_index = first_touch_index + np.random.randint(4,10)
					goal_index = np.clip(goal_index, 0, 80)
					S_goal_list.append(policy_traj_state[goal_index])
					counter += 1
					# print(first_touch_index)
					# print(goal_index)
					# print("S_init")
					# env_set_and_render(env, S_init_list[-1])
					# print("S_goal")
					# import pdb; pdb.set_trace()
					# env_set_and_render(env, S_goal_list[-1])
					
						
				#else: do nothing, don't use this trajectory
		#save to args.save_path
		with open(args.save_path, 'wb') as handle:
			save_dict = {'S_init': S_init_list, 'S_goal':S_goal_list}
			pickle.dump(save_dict, handle, pickle.HIGHEST_PROTOCOL)