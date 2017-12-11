'''
Author: Dian Chen
Note NoRL is the temporal name...
'''

from rllab.

class NoRL(object):
	def __init__(self, 
		env, 
		sup_callback,
		num_timestep_per_stage=1000000,
		max_lookahead=200,
	):
		self.env = env
		self.sup_callback = sup_callback
		self.num_timestep_per_stage = num_timestep_per_stage
		self.max_lookahead = max_lookahead

	def train(self):
		