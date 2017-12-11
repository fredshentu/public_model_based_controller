import tensorflow as tf
from os import listdir
import os.path as osp

def read_and_decode(filename_queue, obs_shape):
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(
		  serialized_example,
		  # Defaults are not specified since both keys are required.
		  features={
				  'action': tf.FixedLenFeature([4], tf.float32),
		  })
	
	

	# Convert label from a scalar uint8 tensor to an int32 scalar.
	action = tf.cast(features['action'], tf.float32)
	
	return action


def inputs(filenames, obs_shape, train=True, batch_size=256, num_epochs = None):
	"""Reads input data num_epochs times.
	Args:
		train: Selects between the training (True) and validation (False) data.
		batch_size: Number of examples per returned batch.
		num_epochs: Number of times to read the input data, or 0/None to
		   train forever.
	Returns:
		A tuple (images, labels), where:
		* images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
		  in the range [-0.5, 0.5].
		* labels is an int32 tensor with shape [batch_size] with the true label,
		  a number in the range [0, mnist.NUM_CLASSES).
		Note that an tf.train.QueueRunner is added to the graph, which
		must be run using e.g. tf.train.start_queue_runners().
	"""

	with tf.name_scope('input'):
		filename_queue = tf.train.string_input_producer(
				filenames, num_epochs=num_epochs)
				
		action = read_and_decode(filename_queue, obs_shape)

		if train:
			num_thread = 1
			queue_capacity = 40000
		else:
			num_thread = 1
			queue_capacity = 4000
		actions = tf.train.batch([ action],\
										batch_size = batch_size, num_threads = num_thread,\
										capacity = queue_capacity, enqueue_many =False)

		
		return  actions

DATA_PATH = "/home/fredshentu/Desktop/dynamic_model_planner_data/Box3dPush-v0_policy_data"
dataset_names = list(map(lambda file_name: osp.join(DATA_PATH, file_name), listdir(DATA_PATH)))
train_set_names = dataset_names
print(dataset_names)
train_action = inputs(train_set_names, 21, train=True)
sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
for i in range(100):
    print("read dataset from tfrecord, index {}".format(i))
    print(sess.run(train_action))
coord.join(threads)
sess.close()
