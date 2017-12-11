import argparse
import joblib
import uuid

import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
from railrl.misc.pyhelper_fns.vis_utils import MyAnimationMulti
import gym

filename = str(uuid.uuid4())

def env_set_state(env, ob):
	qpos_idx = env.env.init_qpos.shape[0]
	env.env.set_state(ob[:qpos_idx], ob[qpos_idx:])
	return 
def env_get_im(env):
	return env.env._render(mode='rgb_array')


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
	parser.add_argument('--visualize', action='store_true', help='use vis tool to visualize directly')
	
	args = parser.parse_args()
	env = gym.make(args.env_name)

	import tensorflow as tf
	from planner import ClippedSgdForwardModelPlanner, InverseModelPlanner, \
					ConstrainedForwardModelPlanner, SgdForwardModelPlanner,\
					FastClippedSgdForwardModelPlanner, FastClippedSgdShootingForwardModelPlanner
	ForwardModelPlanner = FastClippedSgdForwardModelPlanner
	if args.visualize:
		vis_tool = MyAnimationMulti(None, numPlots=4, isIm=[1,1,1,0], axTitles=["S_init", "S_goal", \
					"S_current", "L2 distance S_goal S_current"])
	with open(args.test_set_path, 'rb') as handle:
		data = pickle.load(handle)
		S_init_state_list = data['S_init_state']
		S_init_pixel_list = data['S_init_pixel']
		S_goal_state_list = data['S_goal_state']
		S_goal_pixel_list = data['S_goal_pixel']
	with tf.Session() as sess:
		##load dynamic model
		data = joblib.load(args.dynamic_model_path)
		encoder = data['encoder']
		inverse_model = data['inverse_model']
		forward_model = data['forward_model']
		if args.shooting:
			ForwardModelPlanner = FastClippedSgdShootingForwardModelPlanner
		controller = ForwardModelPlanner(forward_model,encoder, env, sess = sess, pos_only = True)
		env.reset()
		
		for j in range(20):
			S_init_Im = S_init_pixel_list[j]
			S_goal_Im = S_goal_pixel_list[j]
			S_init = S_init_state_list[j]
			S_goal = S_goal_state_list[j]
			
			
			L2_dis_list = []
			Predicted_L2_dis_list = []
			S_current_im_list = []
			

			
			
			env_set_state(env,S_init)
			S_current_im_list.append(env_get_im(env))
			if args.env_name == 'Box3dPush-v1':
				S_goal[4:7] = S_goal[4:7]*10
				S_init[4:7] = S_init[4:7]*10
			obs = S_init
			print("l2 distance arm current and goal:{}".format(np.sum(np.square(S_goal[:4] - obs[:4]))))
			print("l2 distance box current and goal:{}".format(np.sum(np.square(S_goal[4:7] - obs[4:7]))))
			num_iter = 20
			# import pdb; pdb.set_trace()
			L2_dis_list.append(np.sum(np.square(S_goal[:6] - obs[:6])))
			Predicted_L2_dis_list.append(np.sum(np.square(S_goal[:6] - obs[:6])))
			env.render()
			time.sleep(0.1)
			for i in range(num_iter):
				if args.shooting:
					#if steps = min()
					action, predicted_l2_loss= controller.get_action(obs, S_goal, steps = 3)
				else:
					action,predicted_l2_loss = controller.get_action(obs, S_goal)
				Predicted_L2_dis_list.append(predicted_l2_loss)
				obs, r, d, _ = env.step(action)
				print("(S1-S^1)^2 is {}".format(np.sum(np.square(obs[:6]-predicted_l2_loss[2][0][:6]))))
				S_current_im_list.append(env_get_im(env))
				if args.visualize:
					vis_tool._display([S_init_Im, S_goal_Im, S_current_im_list[-1], [L2_dis_list]])
				env.render()
				print("predicted_l2_loss current and goal{}".format(np.sum(predicted_l2_loss[:2])))
				print("l2 distance arm current and goal:{}".format(np.sum(np.square(S_goal[:4] - obs[:4]))))
				print("l2 distance box current and goal:{}".format(np.sum(np.square(S_goal[4:7] - obs[4:7]))))
				L2_dis_list.append(np.sum(np.square(S_goal[:6] - obs[:6])))
			
			if args.store_file is not None:
				with open(args.store_file + "/{}.pickle".format(j), "wb") as handle:
					save_dict={'S_init':S_init_Im, 'S_goal_Im':S_goal_Im,\
										'S_current_im_list': S_current_im_list,\
										'L2_dis_list':L2_dis_list,\
										'predicted_L2_dis_list': Predicted_L2_dis_list,\
					}
					
					pickle.dump(save_dict, handle, pickle.HIGHEST_PROTOCOL)