import numpy
from PIL import Image
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def rgb_to_cieluv(rgb_image):
	# https://docs.opencv.org/3.1.0/de/d25/imgproc_color_conversions.html
	# rgb_image is the matrix of h*w*3
	# representing red, blue and green channel
	_x, _y, _z = numpy.shape(rgb_image)

	# assert that _z is 3
	luv_image = numpy.zeros(shape = (_x, _y, _z))

	for i in range(_x):
		for j in range(_y):
			r, g, b = normalize(rgb_image[i][j][0], rgb_image[i][j][1], rgb_image[i][j][2])
			x, y, z = rgb_to_xyz_pixel(r, g, b)
			l, u, v = xyz_to_luv_pixel(x, y, z)
			# normalize l, u, v
			luv_image[i][j][0] = l*255.0/100.0
			luv_image[i][j][1] = (u+134)*255.0/354.0
			luv_image[i][j][2] = (v+140)*255.0/262.0

	return luv_image

def rgb_to_xyz_pixel(r, g, b):
	# https://docs.opencv.org/3.1.0/de/d25/imgproc_color_conversions.html
	# assert r, g, b are in between 0 to 1, normalized
	x = 0.412453*r + 0.357580*g + 0.180423*b
	y = 0.212671*r + 0.715160*g + 0.072169*b
	z = 0.019334*r + 0.119193*g + 0.950227*b

	return x, y, z

def xyz_to_luv_pixel(x, y, z):
	# https://docs.opencv.org/3.1.0/de/d25/imgproc_color_conversions.html
	if y > 0.008856:
		l = 116*pow(y, 1/3)
	else:
		l = 903.3*y

	u_n = 0.19793943
	v_n = 0.46831096

	u_dash = (4*x)/(x + 15*y + 3*z)
	v_dash = (9*y)/(x + 15*y + 3*z)

	u = 13*l*(u_dash - u_n)
	v = 13*l*(v_dash - v_n)

	# l ranges from 0 to 100
	# u ranges from -134 to 220
	# v ranges from -140 to 122
	return l, u, v

def normalize(r, g, b):
	nr = r/255.0
	ng = g/255.0
	nb = b/255.0

	return nr, ng, nb


'''
def cnn_model_fn(features, labels, mode):
	"""Model function for CNN."""
	# Input Layer
	input_layer = tf.reshape(features["x"], [-1, 224, 224, 1])
	padded_input = tf.pad(input_layer, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
	# Low level feature network
  	
	# Convolutional Layer #1
	conv1 = tf.layers.conv2d(
		inputs=padded_input,
	  	filters=64,
	  	kernel_size=[3, 3],
	  	strides=[1,2,2,1],
	  	padding="VALID",
	  	activation=tf.nn.relu)

  	
  	padded_conv1 = tf.pad(conv1, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
  	# Convolutional Layer #2 
  	conv2 = tf.layers.conv2d(
		inputs=padded_conv1,
	  	filters=128,
	  	kernel_size=[3, 3],
	  	strides=[1,1,1,1],
	  	padding="VALID",
	  	activation=tf.nn.relu)

  	padded_conv2 = tf.pad(conv2, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")

  	# Convolutional Layer #3
  	conv3 = tf.layers.conv2d(
		inputs=padded_conv2,
	  	filters=128,
	  	kernel_size=[3, 3],
	  	strides=[1,2,2,1],
	  	padding="VALID",
	  	activation=tf.nn.relu)

  	padded_conv3 = tf.pad(conv3, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")

  	# Convolutional Layer #4
  	conv4 = tf.layers.conv2d(
		inputs=padded_conv3,
	  	filters=256,
	  	kernel_size=[3, 3],
	  	strides=[1,1,1,1],
	  	padding="VALID",
	  	activation=tf.nn.relu)

  	padded_conv4 = tf.pad(conv4, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")

  	# Convolutional Layer #5
  	conv5 = tf.layers.conv2d(
		inputs=padded_conv4,
	  	filters=256,
	  	kernel_size=[3, 3],
	  	strides=[1,2,2,1],
	  	padding="VALID",
	  	activation=tf.nn.relu)

	padded_conv5 = tf.pad(conv5, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")

  	# Convolutional Layer #6
  	conv6 = tf.layers.conv2d(
		inputs=padded_conv5,
	  	filters=512,
	  	kernel_size=[3, 3],
	  	strides=[1,1,1,1],
	  	padding="VALID",
	  	activation=tf.nn.relu)
  	
  	padded_conv6 = tf.pad(conv6, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")


  	#Global feature network

  	conv_g_1 = tf.layers.conv2d(
		inputs=padded_conv6,
	  	filters=512,
	  	kernel_size=[3, 3],
	  	strides=[1,2,2,1],
	  	padding="VALID",
	  	activation=tf.nn.relu)

  	padded_conv_g_1 = tf.pad(conv_g_1, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")

  	conv_g_2 = tf.layers.conv2d(
		inputs=padded_conv_g_1,
	  	filters=512,
	  	kernel_size=[3, 3],
	  	strides=[1,1,1,1],
	  	padding="VALID",
	  	activation=tf.nn.relu)

  	padded_conv_g_2 = tf.pad(conv_g_2, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")

  	conv_g_3 = tf.layers.conv2d(
		inputs=padded_conv_g_2,
	  	filters=512,
	  	kernel_size=[3, 3],
	  	strides=[1,2,1,1],
	  	padding="VALID",
	  	activation=tf.nn.relu)

  	padded_conv_g_3 = tf.pad(conv_g_3, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")

  	conv_g_4 = tf.layers.conv2d(
		inputs=padded_conv_g_3,
	  	filters=512,
	  	kernel_size=[3, 3],
	  	strides=[1,1,1,1],
	  	padding="VALID",
	  	activation=tf.nn.relu)

  	# Dense Layer 1
  	pool2_flat = tf.reshape(conv_g_4, [-1, 7 * 7 * 512])
  	dense_g_1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  	dropout_g_1 = tf.layers.dropout(
		inputs=dense_g_1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  	# dense layer 2
  	dense_g_2 = tf.layers.dense(inputs=dropout_g_1, units=512, activation=tf.nn.relu)
  	dropout_g_2 = tf.layers.dropout(
		inputs=dense_g_2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)


  	# dense layer 3
  	dense_g_3 = tf.layers.dense(inputs=dropout_g_2, units=256, activation=tf.nn.relu)
  	dropout_g_3 = tf.layers.dropout(
		inputs=dense_g_3, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)



  	#MID LEVEL NETWORK

  	conv_m_1 = tf.layers.conv2d(
		inputs=padded_conv6,
	  	filters=512,
	  	kernel_size=[3, 3],
	  	strides=[1,1,1,1],
	  	padding="VALID",
	  	activation=tf.nn.relu)

  	padded_conv_m_1 = tf.pad(conv_m_1, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")

  	conv_m_2 = tf.layers.conv2d(
		inputs=padded_conv_m_1,
	  	filters=256,
	  	kernel_size=[3, 3],
	  	strides=[1,1,1,1],
	  	padding="VALID",
	  	activation=tf.nn.relu)


  	#FUSION LAYER (YET TO BE MADE) CANT UNDERSTAND HOW TO DO IT
  	fusion = conv_m_2
  	#COLORIZATION lAYER
  	padded_conv_fusion = tf.pad(fusion, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")

  	conv_col_1 = tf.layers.conv2d(
		inputs=padded_conv_fusion,
	  	filters=128,
	  	kernel_size=[3, 3],
	  	strides=[1,1,1,1],
	  	padding="VALID",
	  	activation=tf.nn.relu)

  	predictions = {
		# Generate predictions (for PREDICT and EVAL mode)
	  	"classes": tf.argmax(input=logits, axis=1),
	  	# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
	  	# `logging_hook`.
	  	"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

  	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  	# Calculate Loss (for both TRAIN and EVAL modes)
  	loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  	# Configure the Training Op (for TRAIN mode)
  	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(
		loss=loss,
		global_step=tf.train.get_global_step())
	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
	    "accuracy": tf.metrics.accuracy(
		labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
	    mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def cnn_model_fn(features, labels, mode):
	input_layer = tf.reshape(features["x"], [-1, 224, 224, 1])
	# Convolutional Layer #1
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
	  	filters=64,
	  	kernel_size=[3, 3],
	  	strides=[1,1,1,1],
	  	padding="SAME",
	  	activation=tf.nn.relu)

	# Convolutional Layer #2
	conv2 = tf.layers.conv2d(
		inputs=conv1,
	  	filters=64,
	  	kernel_size=[3, 3],
	  	strides=[1,1,1,1],
	  	padding="SAME",
	  	activation=tf.nn.relu)
	# will go as an input to concat layer

	pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

	# Convolutional Layer #3
	conv3 = tf.layers.conv2d(
		inputs=pool1,
	  	filters=128,
	  	kernel_size=[3, 3],
	  	strides=[1,1,1,1],
	  	padding="SAME",
	  	activation=tf.nn.relu)

	# Convolutional Layer #4
	conv4 = tf.layers.conv2d(
		inputs=conv3,
	  	filters=128,
	  	kernel_size=[3, 3],
	  	strides=[1,1,1,1],
	  	padding="SAME",
	  	activation=tf.nn.relu)
	# will go as input to elemwise sum 1
	
	pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

	# Convolutional Layer #5
	conv5 = tf.layers.conv2d(
		inputs=pool2,
	  	filters=256,
	  	kernel_size=[3, 3],
	  	strides=[1,1,1,1],
	  	padding="SAME",
	  	activation=tf.nn.relu)

	# Convolutional Layer #6
	conv6 = tf.layers.conv2d(
		inputs=conv5,
	  	filters=256,
	  	kernel_size=[3, 3],
	  	strides=[1,1,1,1],
	  	padding="SAME",
	  	activation=tf.nn.relu)
	
	# Convolutional Layer #7
	conv7 = tf.layers.conv2d(
		inputs=conv6,
	  	filters=256,
	  	kernel_size=[3, 3],
	  	strides=[1,1,1,1],
	  	padding="SAME",
	  	activation=tf.nn.relu)
	# will go as input to elemwise sum 2
	
	pool3 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=2)

	# Convolutional Layer #8
	conv8 = tf.layers.conv2d(
		inputs=pool3,
	  	filters=512,
	  	kernel_size=[3, 3],
	  	strides=[1,1,1,1],
	  	padding="SAME",
	  	activation=tf.nn.relu)

	# Convolutional Layer #9
	conv9 = tf.layers.conv2d(
		inputs=conv8,
	  	filters=512,
	  	kernel_size=[3, 3],
	  	strides=[1,1,1,1],
	  	padding="SAME",
	  	activation=tf.nn.relu)
	
	# Convolutional Layer #10
	conv10 = tf.layers.conv2d(
		inputs=conv9,
	  	filters=512,
	  	kernel_size=[3, 3],
	  	strides=[1,1,1,1],
	  	padding="SAME",
	  	activation=tf.nn.relu)
	
	conv11 = tf.layers.conv2d(
		inputs=conv10,
	  	filters=256,
	  	kernel_size=[1, 1],
	  	strides=[1,1,1,1],
	  	padding="SAME",
	  	activation=tf.nn.relu)
	# will go as input to elemwise sum 2
	#will go to concat layer
	
	elemwise2= tf.add(conv7,conv11)

	conv12 = tf.layers.conv2d(
		inputs=elemwise2,
	  	filters=128,
	  	kernel_size=[3, 3],
	  	strides=[1,1,1,1],
	  	padding="SAME",
	  	activation=tf.nn.relu)
	# will go as input to elemwise sum 1
	# will go to concat layer

	elemwise1 = tf.add(conv4,conv12)

	conv13 = tf.layers.conv2d(
		inputs=elemwise1,
	  	filters=64,
	  	kernel_size=[3, 3],
	  	strides=[1,1,1,1],
	  	padding="SAME",
	  	activation=tf.nn.relu)

	#will go to concat layer

	concat = tf.concat([conv2,conv11,conv12,conv13],-1)

	#post concat decoder network
	conv14 = tf.layers.conv2d(
		inputs=concat,
	  	filters=256,
	  	kernel_size=[3, 3],
	  	strides=[1,1,1,1],
	  	padding="SAME",
	  	activation=tf.nn.relu)

	conv15 = tf.layers.conv2d(
		inputs=conv14,
	  	filters=64,
	  	kernel_size=[3, 3],
	  	strides=[1,1,1,1],
	  	padding="SAME",
	  	activation=tf.nn.relu)

	conv16 = tf.layers.conv2d(
		inputs=conv15,
	  	filters=64,
	  	kernel_size=[3, 3],
	  	strides=[1,1,1,1],
	  	padding="SAME",
	  	activation=tf.nn.relu)
	
	#for u value
	conv_u = tf.layers.conv2d(
		inputs=conv16,
	  	filters=50,
	  	kernel_size=[1, 1],
	  	strides=[1,1,1,1],
	  	padding="SAME",
	  	activation=tf.nn.relu)
	final_u = tf.nn.softmax(conv_u)

    
    #for u value
	conv_v = tf.layers.conv2d(
		inputs=conv16,
	  	filters=50,
	  	kernel_size=[1, 1],
	  	strides=[1,1,1,1],
	  	padding="SAME",
	  	activation=tf.nn.relu)
	final_v = tf.nn.softmax(conv_u)
'''

img = Image.open('test.jpg')
rgb_matrix = numpy.array(img)
luv_matrix = rgb_to_cieluv(rgb_matrix)
print(luv_matrix)