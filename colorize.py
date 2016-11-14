import sys
import os
sys.path.append(os.path.join(sys.path[0],'model'))
sys.path.append(os.path.join(sys.path[0],'utils'))
import glob
import feed_input as fi
import convert_images as ci
from batchnorm import BatchNormalizer
import architecture as arch
import tensorflow as tf
from matplotlib import pyplot as plt

filenames = sorted(glob.glob("../Images/*/*.jpg"))
batch_size = 10
num_epochs = 1e+9

#global_step = tf.Variable(0, name='global_step', trainable=False)
phase_train = tf.placeholder(tf.bool, name='phase_train')

rgb_image = fi.input_pipeline(filenames, batch_size, num_epochs=num_epochs)
yuv_image = ci.rgb_to_yuv(rgb_image)

uv_image = tf.concat(3, [tf.split(3, 3, yuv_image)[1], tf.split(3, 3, yuv_image)[2]])
y_image = tf.split(3, 3, yuv_image)[0]

grayscale = tf.concat(3, [y_image, y_image, y_image])


with open("../vgg16.tfmodel", mode = 'rb') as f:
	fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)
tf.import_graph_def(graph_def, input_map={"images": grayscale})
graph = tf.get_default_graph()


with tf.variable_scope('vgg'):
	conv1_2 = arch.batch_normalize(graph.get_tensor_by_name("import/conv1_2/Relu:0"), 64, phase_train)
	conv2_2 = graph.get_tensor_by_name("import/conv2_2/Relu:0")
	conv3_3 = graph.get_tensor_by_name("import/conv3_3/Relu:0")
	conv4_3 = graph.get_tensor_by_name("import/conv4_3/Relu:0")


with tf.variable_scope('uvcolor'):
	weights = {
	
		'wc1' : tf.Variable(tf.truncated_normal([1, 1, 512, 256], mean=0.0, stddev=0.01)),
		'wc2' : tf.Variable(tf.truncated_normal([3, 3, 256, 128], mean=0.0, stddev=0.01)),
		'wc3' : tf.Variable(tf.truncated_normal([3, 3, 128, 64], mean=0.0, stddev=0.01)),
		'wc4' : tf.Variable(tf.truncated_normal([3, 3, 64, 3], mean=0.0, stddev=0.01)),
		'wc5' : tf.Variable(tf.truncated_normal([3, 3, 3, 3], mean=0.0, stddev=0.01)),
		'wc6' : tf.Variable(tf.truncated_normal([3, 3, 3, 2], mean=0.0, stddev=0.01))
	}


_tensors = {
	"conv1_2":		conv1_2,
	"conv2_2":		conv2_2,
	"conv3_3":		conv3_3,
	"conv4_3":		conv4_3,
	"grayscale":	grayscale,
	"weights":		weights
}


uv_output = arch.cnn_model(_tensors, phase_train)
yuv_output = tf.concat(3, [tf.split(3, 3, grayscale)[0], uv_output])
rgb_output = ci.yuv_to_rgb(yuv_output, batch_size)
output = tf.concat(2, [grayscale, rgb_output, rgb_image])


loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.square(tf.sub(rgb_output, rgb_image)),
									reduction_indices=3), reduction_indices=2), reduction_indices=1))


alpha = 1e-4
optimizer = tf.train.AdamOptimizer(alpha)
opt = optimizer.minimize(loss)


init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(init_op)
saver = tf.train.Saver([weights['wc1'], weights['wc2'], weights['wc3'], weights['wc4'],
						weights['wc5'], weights['wc6']])


coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

step = 0
try:
	while not coord.should_stop():
		print step
		step = step + 1
		training_opt = sess.run(opt, feed_dict={phase_train:True})

		if step % 1000 == 0:
			compare_output, cost, pt = sess.run([output, loss, conv1_2], feed_dict={phase_train:False})
			print cost

			if step % 10000 == 0:
				saver.save(sess, 'my-model', global_step=step)

			for j in range(batch_size):
				plt.imsave("../Outputs/image_" +str(step)+"_"+ str(j), compare_output[j])

			sys.stdout.flush()

except Exception as e:
	coord.request_stop(e)
finally:
	coord.request_stop()


coord.join(threads)
sess.close()