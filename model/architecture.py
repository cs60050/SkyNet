import sys
import os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'utils'))
from batchnorm import BatchNormalizer
import tensorflow as tf



def batch_normalize(x, depth, train):
	ewma = tf.train.ExponentialMovingAverage(decay=0.9999)
	epsilon = 1e-3
	norm = BatchNormalizer(depth, epsilon, ewma, True)
	val_check = norm.get_assigner()
	return tf.cond(train, lambda: norm.normalize(x, True), lambda: norm.normalize(x, False))



def conv2d(input, weights, train):
	conv_out = tf.nn.conv2d(input, weights, [1,1,1,1], padding = 'SAME')
	normed_out = batch_normalize(conv_out, weights.get_shape()[3], train)
	return tf.nn.relu(normed_out)



def cnn_model(tensors, train):
	
	norm_features_4_3 = batch_normalize(tensors["conv4_3"], 512, train)
	layer1 = tf.nn.conv2d(norm_features_4_3, tensors["weights"]["wc1"], [1,1,1,1], padding='SAME')
	layer1 = tf.nn.relu(layer1)
	layer1 = tf.image.resize_bilinear(layer1, [56,56])
	norm_features_3_3 = batch_normalize(tensors["conv3_3"], 256, train)
	input1 = tf.add(layer1, norm_features_3_3)

	layer2 = conv2d(input1, tensors["weights"]["wc2"], train)
	layer2 = tf.image.resize_bilinear(layer2, [112,112])
	norm_features_2_2 = batch_normalize(tensors["conv2_2"], 128, train)
	input2 = tf.add(layer2, norm_features_2_2)

	layer3 = conv2d(input2, tensors["weights"]["wc3"], train)
	layer3 = tf.image.resize_bilinear(layer3, [224,224])
	norm_features_1_2 = batch_normalize(tensors["conv1_2"], 64, train)
	input3 = tf.add(layer3, norm_features_1_2)

	layer4 = conv2d(input3, tensors["weights"]["wc4"], train)
	norm_grayscale = batch_normalize(tensors["grayscale"], 3, train)
	input4 = tf.add(layer4, norm_grayscale)

	layer5 = conv2d(input4, tensors["weights"]["wc5"], train)

	layer6 = tf.nn.conv2d(layer5, tensors["weights"]["wc6"], [1,1,1,1], padding='SAME')
	return tf.sigmoid(layer6)