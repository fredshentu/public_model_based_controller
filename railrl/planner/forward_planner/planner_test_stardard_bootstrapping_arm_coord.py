import gym
import argparse
import joblib
# import uuid
import time
import numpy as np
import pickle
from railrl.misc.pyhelper_fns.vis_utils import MyAnimationMulti


# filename = str(uuid.uuid4())

def env_set_state(env, ob):
	qpos_idx = env.env.init_qpos.shape[0]
	env.env.set_state(ob[:qpos_idx], ob[qpos_idx:])
	return 
def env_get_im(env):
	return env.env._render(mode='rgb_array')
def transfer_box_global_np(obs):
	arm2box = obs[4:7]/10.0
	return obs[21:] + arm2box

#testing data is sampled from the pkl file using bootstrapping. Stop whenever time out or distance converged
#every test set should contain 400 S_init, S_goal pairs
if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('dynamic_model_path', type=str,
						help='path to the dynamix_model')
	parser.add_argument('env_name', type = str,
						help='name of env')
	parser.add_argument('test_set_path', type = str,
						help='path to test set')
	parser.add_argument('--store_file', type=str, help="path to store data")
	parser.add_argument('--shooting', action='store_true', help='if using shooting method(need to adjust prediction steps)')
	parser.add_argument('--horizon', type=int, default=1, help='If shooting, num of horizon want to predict')
	parser.add_argument('--variable_horizon', action='store_true')
	parser.add_argument('--vis', action='store_true', help='use vis tool to visualize directly')
	parser.add_argument('--num_test', type=int, default=15, help='number of tests we want to run, every test involves 300 pairs')
	parser.add_argument('--pairs', type=int, default=50, help='number of tests we want to run, every test involves 300 pairs')
	args = parser.parse_args()
	env = gym.make(args.env_name)
	from planner import ClippedSgdForwardModelPlanner, InverseModelPlanner, \
					ConstrainedForwardModelPlanner, SgdForwardModelPlanner,\
					FastClippedSgdForwardModelPlanner, FastClippedSgdShootingForwardModelPlanner, \
					FastClippedSgdShootingForwardModelPlanner_cumulated_obj, CEMPlanner, CEMPlanner_arm_coord
	ForwardModelPlanner = FastClippedSgdForwardModelPlanner
	import tensorflow as tf
	import matplotlib.pyplot as plt
	with open(args.test_set_path, 'rb') as handle:
		data = pickle.load(handle)
		S_init_state_list = data['S_init_state']
		S_init_pixel_list = data['S_init_pixel']
		S_goal_state_list = data['S_goal_state']
		S_goal_pixel_list = data['S_goal_pixel']
		real_steps_list = data['real_steps_list']
		print('#############real step analysis###################')
		print("mean is {} and variance is {}".format(np.mean(real_steps_list), np.var(real_steps_list)))
		# plt.hist(real_steps_list, bins='auto')
		# plt.show()
		# assert(len(S_init_state_list)==400 and len(S_init_pixel_list)==400 and \
		# 			len(S_goal_state_list)==400 and len(S_goal_pixel_list)==400)
	with tf.Session() as sess:
		##load dynamic model
		data = joblib.load(args.dynamic_model_path)
		encoder = data['encoder']
		inverse_model = data['inverse_model']
		forward_model = data['forward_model']
		if args.shooting:
			ForwardModelPlanner = CEMPlanner_arm_coord
		controller = ForwardModelPlanner(forward_model,encoder, env, sess = sess)
		env.reset()
		success_rate = []
		store_first_test_data = True
		if args.vis:
					vis_tool = MyAnimationMulti(None, height = 400, width = 400, numPlots=5, isIm=[1,1,1,0,0], axTitles=["S_init", "S_goal", \
								"S_current", "arm_l2(real_steps:{}) red real, green predicted", "box_l2, red real, green predicted"])
					planner_vis = MyAnimationMulti(None, numPlots = 15, isIm = np.ones(15))
		for j in range(args.num_test):
			#choose the index first
			# index_list = np.random.randint(0,399, size = args.pairs)
			# index_list = [12,34,25]
			index_list = range(0,20)
			succeed_case = 0
			counter = 1
			# import pdb; pdb.set_trace()
			for index in index_list:
				print("start test{}, traj{}".format(j+1, counter))
				now = time.time()
				S_init_state = S_init_state_list[index]
				S_init_pixel = S_init_pixel_list[index]
				S_goal_state = S_goal_state_list[index]
				S_goal_pixel = S_goal_pixel_list[index]
				# real_steps = real_steps_list[index]
				real_steps=-1
				env_set_state(env, S_goal_state)
				planner_S_goal = env.env.get_invariant_obs()
				#get arm_coordinate goal state
				env_set_state(env,S_init_state)
				obs, r, d, env_info = env.step(np.zeros(4))
				obs_arm_coord = env_info['coordinate_transfered_obs']
				#some env requires rescaling.......
				if args.env_name == 'Box3dPush-v1':
					S_goal_state[4:7] = S_goal_state[4:7]*10
					S_init_state[4:7] = S_init_state[4:7]*10
				elif args.env_name == 'Box3dPush-v2':
					S_goal_state[4:7] = S_goal_state[4:7]*10
					S_init_state[4:7] = S_init_state[4:7]*10
					
				arm_l2_dis_list = [np.sum(np.square(S_init_state[:4] - S_goal_state[:4]))]
				arm_predicted_l2_dis_list = [np.sum(np.square(S_init_state[:4] - S_goal_state[:4]))]
				box_l2_dis_list = [np.sum(np.square(S_init_state[4:6] - S_goal_state[4:6]))]
				box_predicted_l2_dis_list = [np.sum(np.square(S_init_state[4:6] - S_goal_state[4:6]))]
				S_current_im_list = [env_get_im(env)]
				
	
	
				planner_input = obs_arm_coord
				# print("l2 distance arm current and goal:{}".format(np.sum(np.square(S_goal[:4] - obs[:4]))))
				# print("l2 distance box current and goal:{}".format(np.sum(np.square(S_goal[4:7] - obs[4:7]))))
				env.render()
				#if the arm can not reach the goal in 50 time steps, timeout 
				
			
				arm_threshold = [0.05]
				box_threshold = [0.01]
				# import pdb; pdb.set_trace()
				for i in range(130):
					if args.shooting:
						#if steps = min(14, num_itr - i) then not work
						#action, predicted_l2_loss= controller.get_action(obs, S_goal, steps = max(14-i, 1))
						if args.variable_horizon:
							action, planner_info = controller.get_action(planner_input, planner_S_goal, steps = max(15-i,1))
						else:
							action, planner_info = controller.get_action(planner_input, planner_S_goal, steps = args.horizon)
					else:
						action, planner_info = controller.get_action(planner_input, planner_S_goal)
					
					print(action)
					if np.isnan(action).any():
						print("WARNING planner crashed; output nan")
						exit()
					#record predicted loss 
					# print("model output action is")
					# print(action)
					arm_predicted_l2_dis = planner_info['arm_loss']
					box_predicted_l2_dis = planner_info['box_loss']
					forward_model_output_list = planner_info['forward_models_outputs']					
					arm_predicted_l2_dis_list.append(arm_predicted_l2_dis)
					box_predicted_l2_dis_list.append(box_predicted_l2_dis)
					# print("arm_loss pred is {} box_loss pred is {}".format(arm_predicted_l2_dis, box_predicted_l2_dis))
					obs, r, d, env_info = env.step(action)
					planner_input = env_info['coordinate_transfered_obs']
					env.render()
					S_current_im_list.append(env_get_im(env))
					arm_l2_dis_list.append(np.sum(np.square(obs[:4] - S_goal_state[:4])))
					box_l2_dis_list.append(np.sum(np.square(obs[4:6] - S_goal_state[4:6])))
					# import pdb; pdb.set_trace()
					if args.vis:
						old_state = obs.copy()
						old_state[4:7] = old_state[4:7]/10.0
						
						im_list = []
						# print(forward_model_output_list)
						for s in forward_model_output_list:
							s = s.flatten()
							
							s[4:7] = transfer_box_global_np(s)
							s = s[:21]
							env_set_state(env, s)
							im_list.append(env_get_im(env))
						env_set_state(env, old_state)
						
						planner_vis._display(im_list)


						arm_threshold.append(0.05)
						box_threshold.append(0.01)
						vis_tool._display([S_init_pixel, S_goal_pixel, S_current_im_list[-1], [range(i+2), arm_l2_dis_list, 'r',\
																		range(i+2), arm_predicted_l2_dis_list, 'g',\
																		range(i+2), arm_threshold, 'b'],\
																		[range(i+2), box_l2_dis_list, 'r',\
																		range(i+2), box_predicted_l2_dis_list, 'g',\
																		range(i+2), box_threshold, 'b']])
					#reaching, only arm pos
					# if arm_l2_dis_list[-1] < 0.05:
					# 	print("traj succeed, at step {}".format(i))
					# 	break
					# print("predicted_l2_loss current and goal{}".format(predicted_l2_loss))
					# print("l2 distance arm current and goal:{}".format(np.sum(np.square(S_goal[:4] - obs[:4]))))
					# print("l2 distance box current and goal:{}".format(np.sum(np.square(S_goal[4:7] - obs[4:7]))))
				if arm_l2_dis_list[-1] < 0.05:
					succeed_case += 1
				if (args.store_file is not None) and store_first_test_data:
					with open(args.store_file + "/{}.pickle".format(counter), "wb") as handle:
						
						save_dict={'S_init_pixel':S_init_pixel, 'S_goal_pixel':S_goal_pixel,\
											'S_current_im_list': S_current_im_list,\
											'arm_l2_dis_list':arm_l2_dis_list,\
											'box_l2_dis_list': box_l2_dis_list,\
											'arm_predicted_l2_dis_list': arm_predicted_l2_dis_list,\
											'box_predicted_l2_dis_list': box_predicted_l2_dis_list,\
											'real_steps':real_steps
						}
						
						pickle.dump(save_dict, handle, pickle.HIGHEST_PROTOCOL)
				counter+=1
				print("time elapsed{}".format(time.time() - now))
			store_first_test_data = False
			print(succeed_case/1.0/args.pairs)
			success_rate.append(succeed_case/1.0/args.pairs)
			
			with open(args.store_file+args.env_name+"success_rate_record.pkl", 'wb') as handle:
				save_dict={"success_record":success_rate}
				pickle.dump(save_dict, handle, pickle.HIGHEST_PROTOCOL)
		success_rate = np.array(success_rate)
		print("###############succeess rate analysis##########################")
		print("mean:{}\n var{}\n".format(np.mean(success_rate), np.sqrt(np.var(success_rate))))
