import sys
import tensorflow as tf

batch_size = 6
num_epochs = 1e+9

def readfile(filename):
	reader = tf.WholeFileReader()
	key,value = reader.read(filename)
	image = tf.image.decode_jpeg(value, channels=3)
	image = tf.image.resize_images(image, 224, 224)
	float_image = tf.div(tf.cast(image,tf.float), 255)
	return float_image

def input_pipeline(filenames, batch_size)
	filename_queue = tf.train.string_input_producer(filenames, num_epochs = num_epochs)
	file = readfile(filename_queue)
	capacity = 200
	min_after_dequeue = 100
	return tf.train.shuffle_batch([file], batch_size=batch_size,
		capacity=capacity, min_after_dequeue=min_after_dequeue)

def rgb_to_yuv(image):
	filter_weights = tf.constant(
    				[[[[0.299, -0.169, 0.499],
    				[0.587, -0.331, -0.418],
    				[0.114, 0.499, -0.0813]]]])
	filter_biases = tf.constant([0.0, 0.5, 0.5])
	out = tf.nn.conv2d(image, filter_weights, [1, 1, 1, 1], 'SAME')
	return tf.nn.bias_add(out, filter_biases)

def yuv_to_rgb(image):
	filter_weights = tf.constant(
					[[[[1.000, 1.000, 1.000],
					[0.000, -0.344, 1.772],
					[1.402, -0.714, 0.000]]]])
	filter_biases = tf.constant([-179.456, 135.459, -226.816])
	scaled_image = tf.mul(image, 255)
	out = tf.nn.conv2d(scaled_image, filter_weights, [1, 1, 1, 1], 'SAME')
	out = tf.nn.bias_add(out, filter_biases)

	temp = tf.maximum(tf.constant(0.0,shape=[batch_size,224,224,3]), out)
	temp = tf.minimum(tf.constant(255.0,shape=[batch_size,224,224,3]), out)
	return tf.div(temp, 255)


