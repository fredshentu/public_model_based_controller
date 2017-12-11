from railrl.misc.pyhelper_fns.vis_utils import MyAnimationMulti
import pickle
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('file', type=str,
						help='path stores the trajectory')
	parser.add_argument('--success_hist', action='store_true', help='show success rate\'s histogram')
	parser.add_argument('--final_dist_hist', action='store_true', help='show final distance\'s histogram')
	parser.add_argument('--time_analysis', action = 'store_true', help='analysis times steps')
	parser.add_argument('--failure_case', action = 'store_true')
	parser.add_argument('--success_case', action = 'store_true')
	args = parser.parse_args()
	if args.success_hist:
		with open(args.file + '/Box3dReachTable-v1success_rate_record.pkl', 'rb') as handle:
			data = pickle.load(handle)['success_record']
		print("success_rate mean:{}, std:{}".format(np.mean(data), np.sqrt(np.var(data))))
		plt.hist(data, bins='auto')
		plt.title('success rate histogram')
		
		plt.show()
		exit()
	elif args.final_dist_hist:
		dist_list = []
		for i in range(50):
			with open(args.file + "/{}.pickle".format(i+1), "rb") as handle:
				data = pickle.load(handle)
				arm_l2_dis_list = data['arm_l2_dis_list']
				box_l2_dis_list = data['box_l2_dis_list']
			dist_list.append(arm_l2_dis_list[-1] + box_l2_dis_list[-1])
		print("#######fianl distance analysis #############")
		print("mean:{}, std{}".format(np.mean(dist_list), np.sqrt(np.var(dist_list))))
		plt.hist(dist_list, bins=100)
		plt.title('fianl_distance histogram')
		plt.show()
		exit()
	elif args.time_analysis:
		step_list = []
		for i in range(50):
			with open(args.file + "/{}.pickle".format(i+1), "rb") as handle:
				data = pickle.load(handle)
				arm_l2_distance = data['arm_l2_dis_list'][-1]
				controller_steps = len(data['arm_l2_dis_list']) - 1
				real_steps = data['real_steps']
			if (arm_l2_distance < 0.05):
				step_list.append(controller_steps - real_steps)
		print("#######fianl distance analysis #############")
		print("success_number {}, mean:{}, std{}".format(len(step_list), np.mean(step_list), np.var(step_list)))
		plt.hist(step_list, bins=100)
		plt.title('controller_time - real_time histogram')
		plt.show()
		exit()
	
	##load data, extract some info first:
	for i in range(50):
		with open(args.file + "{}.pickle".format(i+1), "rb") as handle:
			data = pickle.load(handle)
			arm_l2_dis_list = data["arm_l2_dis_list"]
			final_distance = arm_l2_dis_list[-1]
			if args.success_case and (final_distance < 0.05):
				#video
				
				S_init_pixel = data['S_init_pixel']
				S_goal_pixel = data['S_goal_pixel']
				S_current_im_list = data['S_current_im_list']
				arm_predicted_l2_dis_list = data['arm_predicted_l2_dis_list']
				box_l2_dis_list = data['box_l2_dis_list']
				box_predicted_l2_dis_list = data['box_predicted_l2_dis_list']
				real_step = data['real_steps']
				print("#############real step is {}###############".format(real_step))
				vis_tool = MyAnimationMulti(None, height = 200, width = 200, numPlots=5, isIm=[1,1,1,0,0], axTitles=["S_init", "S_goal", \
								"S_current", "arm_l2 red real, green predicted, real step:{}".format(real_step), \
								"box_l2, red real, green predicted"], large_figure = True)
				arm_threshold=[]
				box_threshold=[]
				for j in range(len(S_current_im_list)):
					arm_threshold.append(0.05)
					box_threshold.append(0.01)
					vis_tool._display([S_init_pixel, S_goal_pixel, S_current_im_list[j], [range(j+1), arm_l2_dis_list[:j+1], 'r',\
																		range(j+1), arm_predicted_l2_dis_list[:j+1], 'g',\
																		range(j+1), arm_threshold, 'b'],\
																		[range(j+1), box_l2_dis_list[:j+1], 'r',\
																		range(j+1), box_predicted_l2_dis_list[:j+1], 'g',\
																		range(j+1), box_threshold, 'b']])
				vis_tool.__del__()
			elif args.failure_case and (final_distance > 0.05):
				#video
				S_init_pixel = data['S_init_pixel']
				S_goal_pixel = data['S_goal_pixel']
				S_current_im_list = data['S_current_im_list']
				arm_predicted_l2_dis_list = data['arm_predicted_l2_dis_list']
				box_l2_dis_list = data['box_l2_dis_list']
				box_predicted_l2_dis_list = data['box_predicted_l2_dis_list']
				real_step = data['real_steps']
				print("#############real step is {}###############".format(real_step))
				vis_tool = MyAnimationMulti(None, height = 200, width = 200, numPlots=5, isIm=[1,1,1,0,0], axTitles=["S_init", "S_goal", \
								"S_current", "arm_l2 red real, green predicted, real step:{}".format(real_step), \
								"box_l2, red real, green predicted"], large_figure = True)
				arm_threshold=[]
				box_threshold=[]
				for j in range(len(S_current_im_list)):
					arm_threshold.append(0.05)
					box_threshold.append(0.01)
					vis_tool._display([S_init_pixel, S_goal_pixel, S_current_im_list[j], [range(j+1), arm_l2_dis_list[:j+1], 'r',\
																		range(j+1), arm_predicted_l2_dis_list[:j+1], 'g',\
																		range(j+1), arm_threshold, 'b'],\
																		[range(j+1), box_l2_dis_list[:j+1], 'r',\
																		range(j+1), box_predicted_l2_dis_list[:j+1], 'g',\
																		range(j+1), box_threshold, 'b']])
				vis_tool.__del__()
	
	for i in range(50):
		with open(args.file + "{}.pickle".format(i), "rb") as handle:
			data = pickle.load(handle)
			S_init_im = data["S_init_pixel"]
			S_goal_im = data["S_goal_Im"]
			S_current_im_list = data["S_current_im_list"]
			L2_dis_list = data["L2_dis_list"]
			predicted_L2_dist_list = data['predicted_L2_dis_list']
		
		# L2_dis_list = np.array(L2_dis_list)
		# import pdb; pdb.set_trace()
		L2_loss = []
		predicted_L2_loss = []
		threshold = []
		for i in range(len(S_current_im_list)):
			print("current index is{}".format(i))
			L2_loss.append(L2_dis_list[i])
			predicted_L2_loss.append(predicted_L2_dist_list[i])
			threshold.append(0.05)
			vis_tool._display([S_init_im, S_goal_im, S_current_im_list[i], [range(i+1), L2_loss, 'r',\
																		range(i+1), predicted_L2_loss, 'g',\
																		range(i+1), threshold, 'b']])
			time.sleep(0.01)