from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re

from cnn import *
from data_process import DataSet
import time
from datetime import datetime
import os
import sys

class Network:
	def __init__(self,train,params):
		self.train = train
		self.weight_decay = 0.01
		#self.final_conv = self.create_net()

	def create_net(self,data_i):
		conv1 = conv2d('conv1' , data_i, [3,3,1,64], stride=1,wd=self.weight_decay)
		conv2 = conv2d('conv2' , conv1, [3,3,64,64], stride=1,wd=self.weight_decay)
		# will go as an input to concat layer (224*224*64)
		pool1 = maxpool2d('mp_1',conv2,[2, 2], stride=2)
	
		conv3 = conv2d('conv3' , conv2, [3,3,64,128], stride=1,wd=self.weight_decay)
		conv4 = conv2d('conv4' , conv3, [3,3,128,128], stride=1,wd=self.weight_decay)
		# will go as input to elemwise sum 1 (112*112*128)
		pool2 = maxpool2d('mp_2',conv4,[2, 2], stride=2)
		conv5 = conv2d('conv5' , pool2, [3,3,128,256], stride=1,wd=self.weight_decay)
		conv6 = conv2d('conv6' , conv5, [3,3,256,256], stride=1,wd=self.weight_decay)
		conv7 = conv2d('conv7' , conv6, [3,3,256,256], stride=1,wd=self.weight_decay)
		# will go as input to elemwise sum 2 (56*56*256)
		pool3 = maxpool2d('mp_3', conv7,[2, 2], stride=2)
		conv8 = conv2d('conv8' , pool3, [3,3,256,512], stride=1,wd=self.weight_decay)
		conv9 = conv2d('conv9' , conv8, [3,3,512,512], stride=1,wd=self.weight_decay)
		conv10 = conv2d('conv10' , conv9, [3,3,512,512], stride=1,wd=self.weight_decay)
		conv11 = conv2d('conv11' , conv10, [3,3,512,256], stride=1,wd=self.weight_decay)
		# will go as input to elemwise sum 2 (28*28*256)
		#will go to concat layer (28*28*256)
		conv11 = deconv2d('deconv1',conv11,[3,3,256,256],stride=2,wd=self.weight_decay)
		# (56 *56 *256)
		elemwise2= tf.add(conv7,conv11)
		# (56 *56 *256)
		conv12 = conv2d('conv12' , elemwise2, [3,3,256,128], stride=1,wd=self.weight_decay)
		# will go as input to elemwise sum 1 (56*56*128)
		# will go to concat layer
		conv12 = deconv2d('deconv2',conv12,[3,3,128,128],stride=2,wd=self.weight_decay)
		elemwise1 = tf.add(conv4,conv12)
		conv13 = conv2d('conv13' , elemwise1, [3,3,128,64], stride=1,wd=self.weight_decay)
		#conv13 = deconv2d('deconv3',conv13,[3,3,64,128],stride=2,wd=self.weight_decay)
		conv11 = deconv2d('deconv4',conv11,[3,3,256,256],stride=2,wd=self.weight_decay)
		#will go to concat layer (112*112*128)
		#conv2 = (224,224,64)
		#conv13 = (224,224,128)
		#conv11 = (224,224,256)
		concat = tf.concat([conv2,conv11,conv12,conv13],-1)
		conv14 = conv2d('conv14' , concat, [3,3,512,256], stride=1,wd=self.weight_decay)
		conv15 = conv2d('conv15' , conv14, [3,3,256,64], stride=1,wd=self.weight_decay)
		conv16 = conv2d('conv16' , conv15, [3,3,64,64], stride=1,wd=self.weight_decay)
		conv_u = conv2d('conv_u' , conv16, [3,3,64,50], stride=1,wd=self.weight_decay)
		final_u = tf.nn.softmax(conv_u)
		conv_u = conv2d('conv_v' , conv16, [3,3,64,50], stride=1,wd=self.weight_decay)
		final_v = tf.nn.softmax(conv_u)
		
		return final_u,final_v

	def loss(self,final_u,final_v,image_u,image_v):
		final_u = tf.reduce_max(final_u,axis=-1)
		final_v = tf.reduce_max(final_v,axis=-1)
		# image_u_sftmx = tf.nn.softmax(image_u)
		# image_v_sftmx = tf.nn.softmax(image_v)
		def_u = tf.reduce_sum(tf.square(tf.subtract(final_u,image_u)))
		def_v = tf.reduce_sum(tf.square(tf.subtract(final_v,image_v)))
		return def_u+def_v
		