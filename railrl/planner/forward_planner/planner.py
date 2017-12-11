from railrl.data_management.simple_replay_pool import SimpleReplayPool
from railrl.predictors.dynamics_model import FullyConnectedEncoder, InverseModel, ForwardModel
import tensorflow as tf
import time
import numpy as np
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from railrl.misc.pyhelper_fns.vis_utils import MyAnimationMulti

def planner_info(arm_loss, box_loss, forward_models_outputs):
	return {'arm_loss':arm_loss, 'box_loss':box_loss, \
				'forward_models_outputs': forward_models_outputs}
def gather_cols(params, indices, name=None):
    """Gather columns of a 2D tensor.
    Args:
        params: A 2D tensor.
        indices: A 1D tensor. Must be one of the following types: ``int32``, ``int64``.
        name: A name for the operation (optional).
    Returns:
        A 2D Tensor. Has the same type as ``params``.
    """
    with tf.op_scope([params, indices], name, "gather_cols") as scope:
        # Check input
        params = tf.convert_to_tensor(params, name="params")
        indices = tf.convert_to_tensor(indices, name="indices")
        try:
            params.get_shape().assert_has_rank(2)
        except ValueError:
            raise ValueError('\'params\' must be 2D.')
        try:
            indices.get_shape().assert_has_rank(1)
        except ValueError:
            raise ValueError('\'params\' must be 1D.')

        # Define op
        p_shape = tf.shape(params)
        p_flat = tf.reshape(params, [-1])
        i_flat = tf.reshape(tf.reshape(tf.range(0, p_shape[0]) * p_shape[1],
                                       [-1, 1]) + indices, [-1])
        return tf.reshape(tf.gather(p_flat, i_flat),
                          [p_shape[0], -1])
"""
Planner takes two states (S_init and S_goal) and output an action.
Fine Tune is out of the scope of Planner

"""

class Planner(object):
	def __init__(
				self,
				dynamic_model,
				encoder,
				sess
				):
		self.encoder = encoder
		self.dynamic_model = dynamic_model
		self.sess = sess
		##initialize the model.....
	def get_action(S_init, S_goal):
		return None
	
"""
Inverde_model planner should be easy, just return the action
"""
class InverseModelPlanner(object):
	def __init__(
				self,
				dynamic_model,
				env,
				encoder,
				sess = None,
				):
		if sess == None:
			sess =tf.get_default_session()
		self.sess = sess
		
		#re-construct the dynamic model
		self.S_init_ph = tf.placeholder(tf.float32, list(env.observation_space.shape))
		self.S_goal_ph = tf.placeholder(tf.float32, list(env.observation_space.shape))
		encoder1 = encoder.get_weight_tied_copy(observation_input=self.S_init_ph)
		encoder2 = encoder.get_weight_tied_copy(observation_input=self.S_goal_ph)
		
		self.inverse_model = dynamic_model.get_weight_tied_copy(feature_input1=encoder1.output, 
														feature_input2=encoder2.output)
		
	def get_action(self, S_init, S_goal):
		action = self.sess.run(self.inverse_model.output, feed_dict = \
								{self.S_init_ph:S_init, self.S_goal_ph: S_goal})
		return action

"""
ForwardModel planner, optimize action according to this objective:
min_{a} (S_next - S_goal)^2
"""

class CEMPlanner_arm_coord():
	def __init__(
				self,
				dynamic_model,
				encoder,
				env,
				sess = None,
				max_length = 15,
				sample_batch_size = 2000,
				top_k = 200,
				action_penalty=False,
				accumulated_loss = False):
		self.sample_batch_size = sample_batch_size
		self.top_k = top_k
		
		
		self.env = env
		if sess == None:
			sess =tf.get_default_session()
		self.sess = sess
		
		self.max_length = max_length
		self.action_ph = tf.placeholder(tf.float32, [max_length, None, 4])
		self.forward_model_list = []
		#build the recurrent model w.t. the max length
		self.S_init_ph = tf.placeholder(tf.float32, [None, 24])
		self.S_goal_ph = tf.placeholder(tf.float32, [None, 24])
		#only two feature encoders
		self.encoder1 = encoder.get_weight_tied_copy(observation_input=self.S_init_ph)
		self.encoder2 = encoder.get_weight_tied_copy(observation_input=self.S_goal_ph)
		forward_model = dynamic_model.get_weight_tied_copy(feature_input=self.encoder1.output, 
														action_input=self.action_ph[0])
		self.forward_model_list.append(forward_model)
		self.forward_model_output_list = [forward_model.output] #for debug purpose only
		
		for i in range(1,max_length):
			forward_model = dynamic_model.get_weight_tied_copy(feature_input = forward_model.output,\
														action_input = self.action_ph[i])
			self.forward_model_list.append(forward_model)
			self.forward_model_output_list.append(forward_model.output)
														
		## objective 
		def transfer_box_global_tf(obs):
			arm2box = gather_cols(obs, [4,5])/10.0
			return gather_cols(obs, [21,22]) + arm2box
			
		self.objective_list = []
		self.arm_loss_list = []
		self.box_loss_list = []
		self.objective_topk_index_list = []
		current_objective = 0
		
		#objective
		
		for forward_model in self.forward_model_list:
			if accumulated_loss:
				current_objective += tf.reduce_sum(tf.square(transfer_box_global_tf(forward_model.output)-\
										transfer_box_global_tf(self.encoder2.output)), axis = 1)
			else:
				current_objective = tf.reduce_sum(tf.square(transfer_box_global_tf(forward_model.output)-\
										transfer_box_global_tf(self.encoder2.output)), axis = 1)
			self.objective_list.append(current_objective)
			self.arm_loss_list.append(tf.reduce_sum(tf.square(forward_model.output[0][:4] - self.encoder2.output[0][:4])))
			self.box_loss_list.append(tf.reduce_sum(tf.square(transfer_box_global_tf(forward_model.output)-\
										transfer_box_global_tf(self.encoder2.output)))*100)

		if action_penalty:
			for i in range(len(self.objective_list)):
				self.objective_list[i] += tf.reduce_sum(tf.square(self.action_ph),axis = [0,2])*0.5
		
	
	
	def get_action(self, S_init, S_goal, steps = 1, plot_loss = False, debug = False, stop_variance = 0.2, stop_itr = 3, init_batch_size = 50000):
		assert(steps <= self.max_length)
		#fit a multivariable Gaussian
		mean_list = None
		cov_matrix = None
		
		batch_S_init = np.dot(np.ones([init_batch_size, 1]), S_init.reshape(1,-1))
		batch_S_goal = np.dot(np.ones([init_batch_size, 1]), S_goal.reshape(1,-1))
		#CEM
		actions = np.random.rand(self.max_length, init_batch_size, 4)*2 - 1
		objective_list = self.sess.run(self.objective_list[steps-1], feed_dict = {self.action_ph:actions, \
												self.S_init_ph:batch_S_init, self.S_goal_ph:batch_S_goal})
		sorted_index = np.argsort(objective_list)[:self.top_k]
		
		# debug
		# action_pen, objective_debug = self.sess.run([tf.reduce_sum(tf.square(self.action_ph),axis = [0,2])*0.3, self.objective_list[14]], feed_dict = {self.action_ph:actions, \
		# 										self.S_init_ph:batch_S_init, self.S_goal_ph:batch_S_goal})
		
		# import pdb; pdb.set_trace()
		best_actions = actions[:,sorted_index, :]
		trans_best_actions = np.moveaxis(best_actions, 0, 1).reshape(self.top_k, -1)
		cov_matrix = np.cov(trans_best_actions.T)
		mean_list = np.mean(trans_best_actions.T, axis = 1)
		batch_S_init = np.dot(np.ones([self.sample_batch_size, 1]), S_init.reshape(1,-1))
		batch_S_goal = np.dot(np.ones([self.sample_batch_size, 1]), S_goal.reshape(1,-1))
		for i in range(stop_itr-1):
			actions = np.random.multivariate_normal(mean_list, cov_matrix, self.sample_batch_size).reshape(self.sample_batch_size, self.max_length, 4)
			actions = np.moveaxis(actions, 0,1)
			
			objective_list = self.sess.run(self.objective_list[steps-1], feed_dict = {self.action_ph:actions, \
												self.S_init_ph:batch_S_init, self.S_goal_ph:batch_S_goal})
			sorted_index = np.argsort(objective_list)[:self.top_k]
			best_actions = actions[:,sorted_index, :]
			trans_best_actions = np.moveaxis(best_actions, 0, 1).reshape(self.top_k, -1)
			cov_matrix = np.cov(trans_best_actions.T)
			mean_list = np.mean(trans_best_actions.T, axis = 1)
			# import pdb; pdb.set_trace()

		#if debug, visualize all forward model's output 
		best_action = best_actions[:,0,:]
		arm_loss, box_loss,forward_models_outputs, final_objective = self.sess.run([self.arm_loss_list[0], self.box_loss_list[0], \
												self.forward_model_output_list, self.objective_list[steps-1]], \
												{self.action_ph: best_action.reshape(15,1,4), \
												self.S_init_ph:[S_init], self.S_goal_ph:[S_goal]})
		print("final objective")
		print(final_objective)
		# import pdb; pdb.set_trace()
		return best_actions[0,0], {'arm_loss':arm_loss, 'box_loss':box_loss, 'forward_models_outputs':forward_models_outputs[:steps]}



class CEMPlanner():
	def __init__(
				self,
				dynamic_model,
				encoder,
				env,
				sess = None,
				pos_only = True,
				max_length = 15,
				sample_batch_size = 2000,
				top_k = 200,
				action_penalty=False,
				accumulated_loss = False):
		self.sample_batch_size = sample_batch_size
		self.top_k = top_k
		
		
		self.env = env
		if sess == None:
			sess =tf.get_default_session()
		self.sess = sess
		
		self.max_length = max_length
		self.action_ph = tf.placeholder(tf.float32, [max_length, None, 4])
		self.forward_model_list = []
		#build the recurrent model w.t. the max length
		self.S_init_ph = tf.placeholder(tf.float32, [None]+list(env.observation_space.shape))
		self.S_goal_ph = tf.placeholder(tf.float32, [None]+list(env.observation_space.shape))
		#only two feature encoders
		self.encoder1 = encoder.get_weight_tied_copy(observation_input=self.S_init_ph)
		self.encoder2 = encoder.get_weight_tied_copy(observation_input=self.S_goal_ph)
		forward_model = dynamic_model.get_weight_tied_copy(feature_input=self.encoder1.output, 
														action_input=self.action_ph[0])
		self.forward_model_list.append(forward_model)
		self.forward_model_output_list = [forward_model.output] #for debug purpose only
		
		for i in range(1,max_length):
			forward_model = dynamic_model.get_weight_tied_copy(feature_input = forward_model.output,\
														action_input = self.action_ph[i])
			self.forward_model_list.append(forward_model)
			self.forward_model_output_list.append(forward_model.output)
														
		## objective 
		self.objective_list = []
		self.arm_loss_list = []
		self.box_loss_list = []
		self.objective_topk_index_list = []
		current_objective = 0
		if pos_only:
			for forward_model in self.forward_model_list:
				if accumulated_loss:
					current_objective += tf.reduce_sum(tf.square(gather_cols(forward_model.output, [4,5,6])\
													- gather_cols(self.encoder2.output, [4,5,6])), axis = 1)
				else:
					current_objective = tf.reduce_sum(tf.square(gather_cols(forward_model.output, list(range(4,7)))\
													- gather_cols(self.encoder2.output, list(range(4,7)))), axis = 1)
				self.objective_list.append(current_objective)
				self.arm_loss_list.append(tf.reduce_sum(tf.square(forward_model.output[0][:4] - self.encoder2.output[0][:4])))
				self.box_loss_list.append(tf.reduce_sum(tf.square(forward_model.output[0][4:6] - self.encoder2.output[0][4:6])))
		else:
			for forward_model in self.forward_model_list:
				self.objective_list.append(tf.reduce_sum(tf.square(forward_model.output[0] - self.encoder2.output[0])))
				self.arm_loss_list.append(tf.reduce_sum(tf.square(forward_model.output[0][:4] - self.encoder2.output[0][:4])))
				self.box_loss_list.append(tf.reduce_sum(tf.square(forward_model.output[0][4:6] - self.encoder2.output[0][4:6])))
		if action_penalty:
			for i in range(len(self.objective_list)):
				self.objective_list[i] += tf.reduce_sum(tf.square(self.action_ph),axis = [0,2])*0.5
		
	
	
	def get_action(self, S_init, S_goal, steps = 1, plot_loss = False, debug = False, stop_variance = 0.2, stop_itr = 3, init_batch_size = 50000):
		assert(steps <= self.max_length)
		#fit a multivariable Gaussian
		mean_list = None
		cov_matrix = None
		
		batch_S_init = np.dot(np.ones([init_batch_size, 1]), S_init.reshape(1,-1))
		batch_S_goal = np.dot(np.ones([init_batch_size, 1]), S_goal.reshape(1,-1))
		#CEM
		actions = np.random.rand(self.max_length, init_batch_size, 4)*2 - 1
		objective_list = self.sess.run(self.objective_list[steps-1], feed_dict = {self.action_ph:actions, \
												self.S_init_ph:batch_S_init, self.S_goal_ph:batch_S_goal})
		sorted_index = np.argsort(objective_list)[:self.top_k]
		
		#debug
		# action_pen, objective_debug = self.sess.run([tf.reduce_sum(tf.square(self.action_ph),axis = [0,2])*0.3, self.objective_list[14]], feed_dict = {self.action_ph:actions, \
		# 										self.S_init_ph:batch_S_init, self.S_goal_ph:batch_S_goal})
		
		# import pdb; pdb.set_trace()
		best_actions = actions[:,sorted_index, :]
		trans_best_actions = np.moveaxis(best_actions, 0, 1).reshape(self.top_k, -1)
		cov_matrix = np.cov(trans_best_actions.T)
		mean_list = np.mean(trans_best_actions.T, axis = 1)
		batch_S_init = np.dot(np.ones([self.sample_batch_size, 1]), S_init.reshape(1,-1))
		batch_S_goal = np.dot(np.ones([self.sample_batch_size, 1]), S_goal.reshape(1,-1))
		for i in range(stop_itr-1):
			actions = np.random.multivariate_normal(mean_list, cov_matrix, self.sample_batch_size).reshape(self.sample_batch_size, self.max_length, 4)
			actions = np.moveaxis(actions, 0,1)
			
			objective_list = self.sess.run(self.objective_list[steps-1], feed_dict = {self.action_ph:actions, \
												self.S_init_ph:batch_S_init, self.S_goal_ph:batch_S_goal})
			sorted_index = np.argsort(objective_list)[:self.top_k]
			best_actions = actions[:,sorted_index, :]
			trans_best_actions = np.moveaxis(best_actions, 0, 1).reshape(self.top_k, -1)
			cov_matrix = np.cov(trans_best_actions.T)
			mean_list = np.mean(trans_best_actions.T, axis = 1)
			# import pdb; pdb.set_trace()

		#if debug, visualize all forward model's output 
		best_action = best_actions[:,0,:]
		arm_loss, box_loss,forward_models_outputs, final_objective = self.sess.run([self.arm_loss_list[0], self.box_loss_list[0], \
												self.forward_model_output_list, self.objective_list[steps-1]], \
												{self.action_ph: best_action.reshape(15,1,4), \
												self.S_init_ph:[S_init], self.S_goal_ph:[S_goal]})
		print("final objective")
		print(final_objective)
		arm_obj = np.sum(np.square(forward_models_outputs[steps-1][0][:4] - S_goal[:4]))
		box_obj = np.sum(np.square(forward_models_outputs[steps-1][0][4:7] - S_goal[4:7]))
		print('arm objective is {}, box objective is {}'.format(arm_obj, box_obj))
		# import pdb; pdb.set_trace()
		return best_actions[0,0], {'arm_loss':arm_loss, 'box_loss':box_loss, 'forward_models_outputs':forward_models_outputs[:steps]}
		
class FastClippedSgdShootingForwardModelPlanner_cumulated_obj(object):
	def __init__(
				self,
				dynamic_model,
				encoder,
				env,
				init_lr = 0.5,
				sess = None,
				pos_only = False,
				max_length = 15,
				):
		if sess == None:
			sess =tf.get_default_session()
		self.sess = sess
		self.init_lr = init_lr
		
		self.max_length = max_length
		self.action_ph = tf.placeholder(tf.float32, [max_length, 1, 4])
		self.forward_model_list = []
		#build the recurrent model w.t. the max length
		self.S_init_ph = tf.placeholder(tf.float32, list(env.observation_space.shape))
		self.S_goal_ph = tf.placeholder(tf.float32, list(env.observation_space.shape))
		#only two feature encoders
		self.encoder1 = encoder.get_weight_tied_copy(observation_input=[self.S_init_ph])
		self.encoder2 = encoder.get_weight_tied_copy(observation_input=[self.S_goal_ph])
		forward_model = dynamic_model.get_weight_tied_copy(feature_input=self.encoder1.output, 
														action_input=self.action_ph[0])
		self.forward_model_list.append(forward_model)
		for i in range(1,max_length):
			forward_model = dynamic_model.get_weight_tied_copy(feature_input = forward_model.output,\
														action_input = self.action_ph[i])
			self.forward_model_list.append(forward_model)

			
														
		## objective 
		self.objective_list = []
		self.forward_model_loss_list = []
		self.arm_loss_list = []
		self.box_loss_list = []
		objective = 0
		factor = 1
		if pos_only:
			for forward_model in self.forward_model_list:
				factor=factor*0.4
				self.forward_model_loss_list.append(tf.reduce_sum(tf.square(forward_model.output[0][:6] - self.encoder2.output[0][:6])))
				
				objective += factor*tf.reduce_sum(tf.square(forward_model.output[0][:6] - self.encoder2.output[0][:6]))
				self.objective_list.append(objective)
				self.arm_loss_list.append(tf.reduce_sum(tf.square(forward_model.output[0][:4] - self.encoder2.output[0][:4])))
				self.box_loss_list.append(tf.reduce_sum(tf.square(forward_model.output[0][4:6] - self.encoder2.output[0][4:6])))
		else:
			for forward_model in self.forward_model_list:
				objective += tf.reduce_sum(tf.square(forward_model.output[0] - self.encoder2.output[0]))
				self.objective_list.append(objective)
			
		self.action_grad_list = []
		for obj in self.objective_list:
			#those tail term in action_ph will receive 0 gradient
			self.action_grad_list.append(tf.gradients(obj, self.action_ph))
		self.vis_tool = MyAnimationMulti(None, numPlots=2, isIm=[0,0], axTitles=['(S1-S_goal)^2', 'sum(S_i-S_goal)^2'])
		
	def get_action(self, S_init, S_goal, steps = None, plot_loss = False):
		if steps == None:
			steps = 1 #greedy planner
		else:
			assert(steps <= self.max_length)
		action = np.zeros([self.max_length, 1, 4])
		action_grad = self.action_grad_list[steps - 1]
		# TODO: Find a good stop criteria
		now = time.time()
		S1_loss_list = []
		Sn_loss_list = []
		for i in range(0,101):
			feed_dict = {self.S_init_ph:S_init, self.S_goal_ph:S_goal, self.action_ph : action}
			S1_loss, Sn_loss = self.sess.run([self.objective_list[0], self.objective_list[steps-1]], feed_dict=feed_dict)
			S1_loss_list.append(S1_loss)
			Sn_loss_list.append(Sn_loss)
			if plot_loss and i%20 ==0:
				self.vis_tool._display([[range(i+1), S1_loss_list],[range(i+1), Sn_loss_list]])
				
			gradient = np.array(self.sess.run(action_grad, feed_dict = feed_dict)[0])
			if np.isnan(gradient).any():
				action = np.random.rand(self.max_length, 1, 4)-0.5
				print('nan gradient step{}'.format(i))
				import pdb; pdb.set_trace()
			else:
				if np.linalg.norm(gradient) > steps*4:
					gradient = gradient/np.linalg.norm(gradient)*4*steps
					
				action -= gradient/1.0*self.init_lr
				action = np.clip(action, -1, 1)
				
			# if i %200 == 0:
			# 	print("#########Optimizing action#########")
			# 	action_loss, predicted_next_state = self.sess.run([self.objective_list[steps-1], self.forward_model_list[steps-1].output], feed_dict = feed_dict)
			# 	box_loss = np.sum(np.square(predicted_next_state[0][4:6] - S_goal[4:6]))
			# 	arm_loss = np.sum(np.square(predicted_next_state[0][0:4] - S_goal[0:4]))
			# 	print("action_loss(sum_square_error(S_goal, S_next)) is {}, box_loss is {}, arm_loss is {}".format(action_loss, box_loss, arm_loss))
			# 	print("current_action is {}".format(action[0][0]))
			# 	# print("current s_next is {}".format(self.sess.run(self.forward_model.output, feed_dict = feed_dict)))
			# 	print("{} sec elapsed for 50 gradient steps".format(time.time() - now))
			# 	now = time.time()
		return action[0][0], self.sess.run([self.arm_loss_list[0], self.box_loss_list[0], self.forward_model_list[0].output], feed_dict)


class FastClippedSgdShootingForwardModelPlanner(object):
	def __init__(
				self,
				dynamic_model,
				encoder,
				env,
				init_lr = 0.5,
				sess = None,
				pos_only = False,
				max_length = 15,
				):
		self.env = env
		if sess == None:
			sess =tf.get_default_session()
		self.sess = sess
		self.init_lr = init_lr
		
		self.max_length = max_length
		self.action_ph = tf.placeholder(tf.float32, [max_length, 1, 4])
		self.forward_model_list = []
		#build the recurrent model w.t. the max length
		self.S_init_ph = tf.placeholder(tf.float32, list(env.observation_space.shape))
		self.S_goal_ph = tf.placeholder(tf.float32, list(env.observation_space.shape))
		#only two feature encoders
		self.encoder1 = encoder.get_weight_tied_copy(observation_input=[self.S_init_ph])
		self.encoder2 = encoder.get_weight_tied_copy(observation_input=[self.S_goal_ph])
		forward_model = dynamic_model.get_weight_tied_copy(feature_input=self.encoder1.output, 
														action_input=self.action_ph[0])
		self.forward_model_list.append(forward_model)
		self.forward_model_output_list = [forward_model.output]
		for i in range(1,max_length):
			forward_model = dynamic_model.get_weight_tied_copy(feature_input = forward_model.output,\
														action_input = self.action_ph[i])
			self.forward_model_list.append(forward_model)
			self.forward_model_output_list.append(forward_model.output)
			
														
		## objective 
		self.objective_list = []
		self.arm_loss_list = []
		self.box_loss_list = []
		if pos_only:
			for forward_model in self.forward_model_list:
				self.objective_list.append(tf.reduce_sum(tf.square(forward_model.output[0][:6] - self.encoder2.output[0][:6])))
				self.arm_loss_list.append(tf.reduce_sum(tf.square(forward_model.output[0][:4] - self.encoder2.output[0][:4])))
				self.box_loss_list.append(tf.reduce_sum(tf.square(forward_model.output[0][4:6] - self.encoder2.output[0][4:6])))
		else:
			for forward_model in self.forward_model_list:
				self.objective_list.append(tf.reduce_sum(tf.square(forward_model.output[0] - self.encoder2.output[0])))
			
		self.action_grad_list = []
		for obj in self.objective_list:
			#those tail term in action_ph will receive 0 gradient
			self.action_grad_list.append(tf.gradients(obj, self.action_ph))
		self.vis_tool = MyAnimationMulti(None, numPlots=2, isIm=[0,0], axTitles=['(S1-S_goal)^2', '(S_n-S_goal)^2'])
	
	def get_action(self, S_init, S_goal, steps = None, plot_loss = False):
		if steps == None:
			steps = 1 #greedy planner
		else:
			assert(steps <= self.max_length)
		action = np.zeros([self.max_length, 1, 4])
		action_grad = self.action_grad_list[steps - 1]
		# TODO: Find a good stop criteria
		now = time.time()
		S1_loss_list = []
		Sn_loss_list = []
		for i in range(0,51):
			feed_dict = {self.S_init_ph:S_init, self.S_goal_ph:S_goal, self.action_ph : action}
			S1_loss, Sn_loss = self.sess.run([self.box_loss_list[0], self.box_loss_list[steps-1]], feed_dict=feed_dict)
			S1_loss_list.append(S1_loss)
			Sn_loss_list.append(Sn_loss)
			if plot_loss and i %1 == 0:
				self.vis_tool._display([[range(i+1), S1_loss_list],[range(i+1), Sn_loss_list]])
				
			gradient = np.array(self.sess.run(action_grad, feed_dict = feed_dict)[0])
			if np.isnan(gradient).any():
				action = np.random.rand(self.max_length, 1, 4)-0.5
				print('nan gradient step{}'.format(i))
				import pdb; pdb.set_trace()
			else:
				if np.linalg.norm(gradient) > steps*4:
					gradient = gradient/np.linalg.norm(gradient)*4*steps
					
				action -= gradient/(1.+i*0.05)*self.init_lr
				action = np.clip(action, -1, 1)
		arm_loss, box_loss, forward_models_outputs = \
			self.sess.run([self.arm_loss_list[0], self.box_loss_list[0], \
			self.forward_model_output_list], feed_dict)
		return action[0][0], planner_info(arm_loss, box_loss, forward_models_outputs[:steps])
		
		
class FastClippedSgdForwardModelPlanner(object):
	def __init__(
				self,
				dynamic_model,
				encoder,
				env,
				action_initializer = None,
				init_lr = 1,
				sess = None,
				pos_only = False,
				):
		if sess == None:
			sess =tf.get_default_session()
		self.sess = sess
		
		
		
		# with tf.variable_scope('action_optimizer'):
		# 	self.action = tf.get_variable('planner_action', [1] + list(env.action_space.shape), initializer=action_initializer)
		self.action_ph = tf.placeholder(tf.float32, [None, 4])
		self.S_init_ph = tf.placeholder(tf.float32, list(env.observation_space.shape))
		self.S_goal_ph = tf.placeholder(tf.float32, list(env.observation_space.shape))
		self.encoder1 = encoder.get_weight_tied_copy(observation_input=[self.S_init_ph])
		self.encoder2 = encoder.get_weight_tied_copy(observation_input=[self.S_goal_ph])
		self.forward_model = dynamic_model.get_weight_tied_copy(feature_input=self.encoder1.output, 
														action_input=self.action_ph)
														
		## objective 
		if pos_only:
			
			self.objective = tf.reduce_sum(tf.square(self.forward_model.output[0][:6] - self.encoder2.output[0][:6]))
		else:
			self.objective = tf.reduce_sum(tf.square(self.forward_model.output - self.encoder2.output))
			
		self.arm_loss = tf.reduce_sum(tf.square(self.forward_model.output[0][:4] - self.encoder2.output[0][:4]))
		self.box_loss = tf.reduce_sum(tf.square(self.forward_model.output[0][4:6] - self.encoder2.output[0][4:6]))
		#Adam optimizer has its own variables. Wrap it by a namescope
		self.action_grad = tf.gradients(self.objective, self.action_ph)
		# with tf.variable_scope('action_optimizer'):
		# 	self.action_opt = tf.train.AdamOptimizer(init_lr).minimize(self.objective, var_list = [self.clipped_action])
			# self.action_gradient = tf.train.AdamOptimizer(init_lr).compute_gradients(self.objective, var_list = [self.action])

	def get_action(self, S_init, S_goal):
		#first re-initialize everyvariables in "action_optimizer"
		# variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope='action_optimizer')
		# self.sess.run(tf.initialize_variables(variables))
		action = np.random.rand(4)-0.5
		
		# TODO: Find a good stop criteria
		now = time.time()
		
		for i in range(0,151):
			feed_dict = {self.S_init_ph:S_init, self.S_goal_ph:S_goal, self.action_ph : [action]}
			gradient = self.sess.run([self.action_grad], feed_dict = feed_dict)[0][0][0]
			#raises NotImplementedError: ('Trying to optimize unsupported type ', <tf.Tensor 'clip_by_value:0' shape=(1, 4) dtype=float32>)
			#this code does not work....
			# import pdb; pdb.set_trace()
			action -= gradient/(1.+i*0.2)*0.5
			action = np.clip(action, -1, 1)
			if i %50 == 0:
				print("#########Optimizing action#########")
				action_loss = self.sess.run(self.objective, feed_dict = feed_dict)
				print("action_loss(sum_square_error(S_goal, S_next)) is {}".format(action_loss))
				print("current_action is {}".format(action))
				# print("current s_next is {}".format(self.sess.run(self.forward_model.output, feed_dict = feed_dict)))
				print("{} sec elapsed for 50 gradient steps".format(time.time() - now))
				now = time.time()
		return action, self.sess.run([ self.arm_loss, self.box_loss], feed_dict = feed_dict)

class SgdForwardModelPlanner(object):
	def __init__(
				self,
				dynamic_model,
				encoder,
				env,
				action_initializer = None,
				init_lr = 1e-1,
				sess = None,
				pos_only = False,
				):
		if sess == None:
			sess =tf.get_default_session()
		self.sess = sess
		
		##re-construct the model
		if action_initializer is None:
			action_initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
		
		with tf.variable_scope('action_optimizer'):
			self.action = tf.get_variable('planner_action', [1] + list(env.action_space.shape), initializer=action_initializer)
		self.clipped_action = tf.clip_by_value(self.action, -1, 1)
		# import pdb; pdb.set_trace()
		self.S_init_ph = tf.placeholder(tf.float32, list(env.observation_space.shape))
		self.S_goal_ph = tf.placeholder(tf.float32, list(env.observation_space.shape))
		self.encoder1 = encoder.get_weight_tied_copy(observation_input=[self.S_init_ph])
		self.encoder2 = encoder.get_weight_tied_copy(observation_input=[self.S_goal_ph])
		self.forward_model = dynamic_model.get_weight_tied_copy(feature_input=self.encoder1.output, 
														action_input=self.action)
														
		## objective 
		if pos_only:
			
			self.objective = tf.reduce_sum(tf.square(self.forward_model.output[0][:6] - self.encoder2.output[0][:6]))
		else:
			self.objective = tf.reduce_sum(tf.square(self.forward_model.output - self.encoder2.output))
			
		#Adam optimizer has its own variables. Wrap it by a namescope

		with tf.variable_scope('action_optimizer'):
			self.action_opt = tf.train.AdamOptimizer(init_lr).minimize(self.objective, var_list = [self.clipped_action])
			# self.action_gradient = tf.train.AdamOptimizer(init_lr).compute_gradients(self.objective, var_list = [self.action])

	def get_action(self, S_init, S_goal):
		#first re-initialize everyvariables in "action_optimizer"
		variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope='action_optimizer')
		self.sess.run(tf.initialize_variables(variables))
		
		feed_dict = {self.S_init_ph:S_init, self.S_goal_ph:S_goal}
		
		# TODO: Find a good stop criteria
		now = time.time()
		
		for i in range(0,150):
			gradient = self.sess.run([self.action_opt], feed_dict = feed_dict)
			#raises NotImplementedError: ('Trying to optimize unsupported type ', <tf.Tensor 'clip_by_value:0' shape=(1, 4) dtype=float32>)
			#this code does not work....
			
			if i %50 == 0:
				print("#########Optimizing action#########")
				action_loss = self.sess.run(self.objective, feed_dict = feed_dict)
				print("action_loss(sum_square_error(S_goal, S_next)) is {}".format(action_loss))
				print("current_action is {}".format(self.sess.run(self.action)))
				# print("current s_next is {}".format(self.sess.run(self.forward_model.output, feed_dict = feed_dict)))
				print("{} sec elapsed for 50 gradient steps".format(time.time() - now))
				now = time.time()
		return self.sess.run([self.action, self.objective], feed_dict = feed_dict)
		
	#debug API
	def predict_next_state(self, current_state, action, goal_state):
		
		feed_dict = {self.S_init_ph:current_state, self.S_goal_ph: goal_state}
		old_action = self.sess.run(self.action)
		#assign new action
		self.sess.run(self.action.assign([action]))
		
		next_state, S_init, S_goal, loss = self.sess.run([self.forward_model.output,\
													self.encoder1.output,\
													self.encoder2.output,\
													self.objective], feed_dict = feed_dict)
		#assign back the old action
		self.sess.run(self.action.assign(old_action))
		return next_state, S_init, S_goal, loss

class ClippedSgdForwardModelPlanner(object):
	def __init__(
				self,
				dynamic_model,
				encoder,
				env,
				action_initializer = None,
				init_lr = 1e-1,
				sess = None,
				pos_only = False,
				):
		if sess == None:
			sess =tf.get_default_session()
		self.sess = sess
		
		##re-construct the model
		if action_initializer is None:
			action_initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
		
		with tf.variable_scope('action_optimizer'):
			self.action = tf.get_variable('planner_action', [1] + list(env.action_space.shape), initializer=action_initializer)
		self.clipped_action = tf.clip_by_value(self.action, -1, 1)
		self.S_init_ph = tf.placeholder(tf.float32, list(env.observation_space.shape))
		self.S_goal_ph = tf.placeholder(tf.float32, list(env.observation_space.shape))
		self.encoder1 = encoder.get_weight_tied_copy(observation_input=[self.S_init_ph])
		self.encoder2 = encoder.get_weight_tied_copy(observation_input=[self.S_goal_ph])
		self.forward_model = dynamic_model.get_weight_tied_copy(feature_input=self.encoder1.output, 
														action_input=self.action)
														
		## objective 
		if pos_only:
			
			self.objective = tf.reduce_sum(tf.square(self.forward_model.output[0][:6] - self.encoder2.output[0][:6]))
		else:
			self.objective = tf.reduce_sum(tf.square(self.forward_model.output - self.encoder2.output))
			
		#Adam optimizer has its own variables. Wrap it by a namescope

		with tf.variable_scope('action_optimizer'):
			self.action_opt = tf.train.AdamOptimizer(init_lr).minimize(self.objective, var_list = [self.action])
			self.action_gradient = tf.train.AdamOptimizer(init_lr).compute_gradients(self.objective, var_list = [self.action])
		
	
		
	def get_action(self, S_init, S_goal):
		#first re-initialize everyvariables in "action_optimizer"
		variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope='action_optimizer')
		self.sess.run(tf.initialize_variables(variables))
		feed_dict = {self.S_init_ph:S_init, self.S_goal_ph:S_goal}
		
		# TODO: Find a good stop criteria
		now = time.time()
		
		for i in range(0,150):
			#normal speed
			self.sess.run([self.action_opt], feed_dict = feed_dict)
			#slow and will be slower and slower
			# self.sess.run([self.clipped_action, self.action.assign(self.clipped_action), self.action_opt], \
			# 														feed_dict = feed_dict)
			if i %50 == 0:
				print("#########Optimizing action#########")
				action_loss = self.sess.run(self.objective, feed_dict = feed_dict)
				print("action_loss(sum_square_error(S_goal, S_next)) is {}".format(action_loss))
				print("current_action is {}".format(self.sess.run(self.clipped_action)))
				# print("current s_next is {}".format(self.sess.run(self.forward_model.output, feed_dict = feed_dict)))
				print("{} sec elapsed for 100 gradient steps".format(time.time() - now))
				now = time.time()
		return self.sess.run([self.action, self.objective], feed_dict = feed_dict)
		
	#debug API
	def predict_next_state(self, current_state, action, goal_state):
		feed_dict = {self.S_init_ph:current_state, self.S_goal_ph: goal_state}
		old_action = self.sess.run(self.action)
		#assign new action
		self.sess.run(self.action.assign([action]))
		
		next_state, S_init, S_goal, loss = self.sess.run([self.forward_model.output,\
													self.encoder1.output,\
													self.encoder2.output,\
													self.objective], feed_dict = feed_dict)
		#assign back the old action
		self.sess.run(self.action.assign(old_action))
		return next_state, S_init, S_goal, loss


from sandbox.rocky.tf.core.parameterized import Parameterized


class ParameterizedAction(Parameterized):
	def __init__(self, env, sess, action_initializer = None):
		Parameterized.__init__(self)
		
		if action_initializer is None:
			action_initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
			
		with tf.variable_scope('action_optimizer'):
			self.action = tf.get_variable('planner_action', [1] + list(env.action_space.shape), initializer=action_initializer)
		self.sess = sess
		self.env = env
	def get_action(self):
		return self.sess.run(self.action)
	def initalize_action(self):
		self.sess.run(tf.initialize_variables(self.action))
		return 


	
class ConstrainedForwardModelPlanner(object):
	def __init__(
				self,
				dynamic_model,
				encoder,
				env,
				sess = None,
				pos_only = False,
				action_initializer = None,
				optimizer = tf.contrib.opt.ScipyOptimizerInterface,
				):
		if sess == None:
			sess =tf.get_default_session()
		self.sess = sess
		if action_initializer is None:
			action_initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
		
		with tf.variable_scope('action_optimizer'):
			self.action = tf.get_variable('planner_action', [1,4], initializer=action_initializer)
		## rebuild the dynamic model
		self.S_init_ph = tf.placeholder(tf.float32, list(env.observation_space.shape))
		self.S_goal_ph = tf.placeholder(tf.float32, list(env.observation_space.shape))
		self.encoder1 = encoder.get_weight_tied_copy(observation_input=[self.S_init_ph])
		self.encoder2 = encoder.get_weight_tied_copy(observation_input=[self.S_goal_ph])
		self.forward_model = dynamic_model.get_weight_tied_copy(feature_input=self.encoder1.output, 
														action_input=self.action)
														
		## objective 
		if pos_only:
			
			self.objective = tf.reduce_sum(tf.square(self.forward_model.output[0][:6] - self.encoder2.output[0][:6]))
		else:
			self.objective = tf.reduce_sum(tf.square(self.forward_model.output - self.encoder2.output))

		
		self.loss = self.objective

		self.inequalities = []
		for i in range(4):
			self.inequalities.append(1-tf.square(self.action[0][i]))
		
		# Our default SciPy optimization algorithm, L-BFGS-B, does not support
		# general constraints. Thus we use SLSQP instead.
	def get_action(self, S_init, S_goal):
		#first re-initialize everyvariables in "action_optimizer"
		self.sess.run(tf.initialize_variables([self.action]))
		feed_dict = {self.S_init_ph:S_init, self.S_goal_ph:S_goal}
		
		# need to re-initialize optimizer every time want to use it or it will optimize action without enforcing constrains.
		optimizer = tf.contrib.opt.ScipyOptimizerInterface(
			self.loss, var_list = [self.action], inequalities=self.inequalities, method='SLSQP')
		now = time.time()
		
		optimizer.minimize(self.sess, feed_dict = feed_dict)		
		print("it takes {} to optimize the action".format(time.time() - now))
		return self.sess.run([self.action, self.loss], feed_dict = feed_dict)