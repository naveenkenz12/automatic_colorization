from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re
def _variable(name, shape, initializer):
	var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
	return var

def _variable_with_weight_decay(name, shape, stddev, wd):
	var = _variable(name, shape,
		tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
	if wd is not None:
		weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var

def conv2d(scope, input, kernel_size, stride=1, wd=0.001):
	name = scope
	with tf.variable_scope(scope) as scope:
		kernel = _variable_with_weight_decay('weights',shape=kernel_size,stddev=5e-2,wd=wd)
		conv = tf.nn.conv2d(input, kernel, [1, stride, stride, 1], padding='SAME')
		biases = _variable('biases', kernel_size[3:], tf.constant_initializer(0.0))
		bias = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(bias)
	return conv1

def deconv2d(scope, input, kernel_size, stride=1, wd=0.001):
	pad_size = int((kernel_size[0] - 1)/2)
	#input = tf.pad(input, [[0,0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], "CONSTANT")
	batch_size, height, width, in_channel = [int(i) for i in input.get_shape()]
	out_channel = kernel_size[3] 
	kernel_size = [kernel_size[0], kernel_size[1], kernel_size[3], kernel_size[2]]
	output_shape = [batch_size, height * stride, width * stride, out_channel]
	with tf.variable_scope(scope) as scope:
		kernel = _variable_with_weight_decay('weights', shape=kernel_size,stddev=5e-2,wd=wd)
		deconv = tf.nn.conv2d_transpose(input, kernel, output_shape, [1, stride, stride, 1], padding='SAME')
		biases = _variable('biases', (out_channel), tf.constant_initializer(0.0))
		bias = tf.nn.bias_add(deconv, biases)
		deconv1 = tf.nn.relu(bias)

	return deconv1

def maxpool2d(scope, input, pool_size, stride=2, wd=0.001):
	pool2 = tf.layers.max_pooling2d(inputs=input, pool_size=pool_size, strides=stride)
	return pool2

