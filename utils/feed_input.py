import tensorflow as tf

def readfile(filename):
	try:
		reader = tf.WholeFileReader()
		key,value = reader.read(filename)
		image = tf.image.decode_jpeg(value, channels=3)
		image = tf.image.resize_images(image, 224, 224)
		float_image = tf.div(tf.cast(image,tf.float32), 255)
		return float_image
	except:
		print -1
		return readfile(filename)
		


def input_pipeline(filenames, batch_size, num_epochs=None):
	filename_queue = tf.train.string_input_producer(filenames, num_epochs = num_epochs)
	file = readfile(filename_queue)
	capacity = 200
	min_after_dequeue = 100
	return tf.train.shuffle_batch([file], batch_size=batch_size,
		capacity=capacity, min_after_dequeue=min_after_dequeue)