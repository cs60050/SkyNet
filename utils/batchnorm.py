import tensorflow as tf

class BatchNormalizer(object):
	
	def __init__(self, depth, epsilon, ewma, scale):
		self.mean = tf.Variable(tf.constant(0.0, shape=[depth]), trainable=False)
		self.variance = tf.Variable(tf.constant(1.0, shape=[depth]), trainable=False)
		self.beta = tf.Variable(tf.constant(0.0, shape=[depth]))
		self.gamma = tf.Variable(tf.constant(1.0, shape=[depth]))
		self.ewma_trainer = ewma
		self.epsilon = epsilon
		self.scale_after_norm = scale

	def get_assigner(self):
		return self.ewma_trainer.apply([self.mean, self.variance])

	def normalize(self, x, train=True):
		if train:
			mean, variance = tf.nn.moments(x, [0, 1, 2])
			assign_mean = self.mean.assign(mean)
			assign_variance = self.variance.assign(variance)
			with tf.control_dependencies([assign_mean, assign_variance]):
				return tf.nn.batch_normalization(x, mean, variance, self.beta, self.gamma 
					if self.scale_after_norm else None, self.epsilon)
		else:
			mean = self.ewma_trainer.average(self.mean)
			variance = self.ewma_trainer.average(self.variance)
			local_beta = tf.identity(self.beta)
			local_gamma = tf.identity(self.gamma)
			return tf.nn.batch_normalization(x, mean, variance, local_beta, local_gamma 
					if self.scale_after_norm else None, self.epsilon)
