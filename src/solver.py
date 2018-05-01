from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re

from cnn import *
from network import Network
from data_process import DataSet
import time
from datetime import datetime
import os
import sys

class Solver(object):

	def __init__(self, train=True, params=None):
		self.image_size = int(params['image_size'])
		self.height = self.image_size
		self.width = self.image_size
		self.batch_size = int(params['batch_size'])
		self.num_gpus = 1
		self.learning_rate = float(params['learning_rate'])
		self.moment = float(params['moment'])
		self.max_steps = int(params['max_iterators'])
		self.train_dir = str(params['train_dir'])
		self.lr_decay = float(params['lr_decay'])
		self.decay_steps = int(params['decay_steps'])
		self.train = train
		self.net = Network(train=train, params=params)		
		self.dataset = DataSet(params=params)

	def construct_graph(self, scope):
		self.data_l = tf.placeholder(tf.float32, (self.batch_size, self.height, self.width, 1))
		self.gt_ab_313 = tf.placeholder(tf.float32, (self.batch_size, self.height, self.width))
		self.prior_boost_nongray = tf.placeholder(tf.float32, (self.batch_size, self.height, self.width))
		self.final_u,self.final_v = self.net.create_net(self.data_l)
		self.new_loss = self.net.loss(self.final_u,self.final_v, self.prior_boost_nongray, self.gt_ab_313)
		tf.summary.scalar('new_loss', self.new_loss)
		#tf.summary.scalar('total_loss', g_loss)
		return self.new_loss

	def train_model(self):
		self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
		learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
																				 self.decay_steps, self.lr_decay, staircase=True)
		opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta2=0.99)
		with tf.name_scope('gpu') as scope:
			self.new_loss = self.construct_graph(scope)
			self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
		grads = opt.compute_gradients(self.new_loss)
		self.summaries.append(tf.summary.scalar('learning_rate', learning_rate))
		for grad, var in grads:
			if grad is not None:
				self.summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
		apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)

		print("hi")
		for var in tf.trainable_variables():
			self.summaries.append(tf.summary.histogram(var.op.name, var))

		variable_averages = tf.train.ExponentialMovingAverage(
				0.999, self.global_step)
		variables_averages_op = variable_averages.apply(tf.trainable_variables())

		train_op = tf.group(apply_gradient_op, variables_averages_op)
		saver = tf.train.Saver(write_version=1)
		saver1 = tf.train.Saver()
		summary_op = tf.summary.merge(self.summaries)
		init =  tf.global_variables_initializer()
		config = tf.ConfigProto(allow_soft_placement=True)
		config.gpu_options.allow_growth = True
		sess = tf.Session(config=config)
		sess.run(init)
		#saver1.restore(sess, './models/model.ckpt')
		#nilboy
		summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)
		for step in range(self.max_steps):
			print(step)
			start_time = time.time()
			t1 = time.time()
			data_l, gt_ab_313, prior_boost_nongray = self.dataset.batch()
			t2 = time.time()
			_, loss_value = sess.run([train_op,self.new_loss], feed_dict={self.data_l:data_l, self.gt_ab_313:gt_ab_313, self.prior_boost_nongray:prior_boost_nongray})
			duration = time.time() - start_time
			t3 = time.time()
			print('io: ' + str(t2 - t1) + '; compute: ' + str(t3 - t2))
			assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
			# if step % 10 == 0:
			# 	summary_str = sess.run(summary_op, feed_dict={self.data_l:data_l, self.gt_ab_313:gt_ab_313, self.prior_boost_nongray:prior_boost_nongray})
			# 	summary_writer.add_summary(summary_str, step)
			if step % 1000 == 0:
				checkpoint_path = os.path.join(self.train_dir, 'model.ckpt')
				saver.save(sess, checkpoint_path, global_step=step)