from data_process import DataSet
import sys
from network import Network
#from solver import Solver
from utils import *

def construct_graph(self, scope):
	data_l = tf.placeholder(tf.float32, (batch_size, height, width, 1))
	gt_ab_313 = tf.placeholder(tf.float32, (batch_size, height, width))
	prior_boost_nongray = tf.placeholder(tf.float32, (batch_size, height, width))
	final_u,final_v = net.create_net(data_l)
	new_loss = net.loss(final_u,final_v, prior_boost_nongray, gt_ab_313)
	tf.summary.scalar('new_loss', new_loss)
	#tf.summary.scalar('total_loss', g_loss)
	return new_loss

params = {}
#common_params, dataset_params, net_params, solver_params = process_config(conf_file)
params["batch_size"] = 40
params['image_size'] = 224
params['learning_rate'] = 0.00003
params['moment'] = 1
params['max_iterators'] = 100000
params['train_dir'] = '../datasets/'
params['path'] = "../datasets/"
params['lr_decay'] = 0.5
params['decay_steps'] = 10
image_size = int(params['image_size'])
height = image_size
width = image_size
batch_size = int(params['batch_size'])
num_gpus = 1
learning_rate = float(params['learning_rate'])
moment = float(params['moment'])
max_steps = int(params['max_iterators'])
train_dir = str(params['train_dir'])
lr_decay = float(params['lr_decay'])
decay_steps = int(params['decay_steps'])
train = True
net = Network(train=train, params=params)		
dataset = DataSet(params=params)


global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
learning_rate = tf.train.exponential_decay(learning_rate, global_step,decay_steps, lr_decay, staircase=True)
opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta2=0.99)
#print("hello")
with tf.name_scope('entry') as scope:
	new_loss = construct_graph(scope)
	summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
grads = opt.compute_gradients(new_loss)
summaries.append(tf.summary.scalar('learning_rate', learning_rate))
#print("hello1")
for grad, var in grads:
	if grad is not None:
		summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
#print("hi")
for var in tf.trainable_variables():
	summaries.append(tf.summary.histogram(var.op.name, var))
variable_averages = tf.train.ExponentialMovingAverage(0.999, global_step)
variables_averages_op = variable_averages.apply(tf.trainable_variables())
train_op = tf.group(apply_gradient_op, variables_averages_op)
saver = tf.train.Saver(write_version=1)
saver1 = tf.train.Saver()
summary_op = tf.summary.merge(summaries)
init =  tf.global_variables_initializer()
config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)
sess.run(init)
#saver1.restore(sess, './models/model.ckpt')
summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
for step in range(max_steps):
	print(step)
	start_time = time.time()
	t1 = time.time()
	data_l, gt_ab_313, prior_boost_nongray = dataset.batch()
	t2 = time.time()
	_, loss_value = sess.run([train_op,new_loss], feed_dict={data_l:data_l, gt_ab_313:gt_ab_313, prior_boost_nongray:prior_boost_nongray})
	duration = time.time() - start_time
	t3 = time.time()
	print('io: ' + str(t2 - t1) + '; compute: ' + str(t3 - t2))
	assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
	# if step % 10 == 0:
	# 	summary_str = sess.run(summary_op, feed_dict={data_l:data_l, gt_ab_313:gt_ab_313, prior_boost_nongray:prior_boost_nongray})
	# 	summary_writer.add_summary(summary_str, step)
	if step % 1000 == 0:
		checkpoint_path = os.path.join(train_dir, 'model.ckpt')
		saver.save(sess, checkpoint_path, global_step=step)