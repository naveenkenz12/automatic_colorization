import numpy
from PIL import Image
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
class DataSet:
	
	def __init__(self,params):
		self.image_size = int(params['image_size'])
      	self.batch_size = int(params['batch_size'])
        self.data_path = str(params['path'])
    	self.image_queue = Queue(maxsize=50000)
	    self.batch_queue = Queue(maxsize=10000)
	    self.record_list = []  
	    #code to create record list
    	'''
    	for loops to read all files'''
    	# record list contain names of all training images
    	for ele in record_list:
	    	img = Image.open(ele)
			out = numpy.array(img)
	      	if len(out.shape)==3 and out.shape[2]==3:
	        	self.image_queue.put(out)
	    while True:
      		images = []
      		for i in range(self.batch_size):
        		image = self.image_queue.get()
        		image = self.rgb_to_cieluv(image)
        		images.append(image)
      		images = np.asarray(images, dtype=np.uint8)
			self.batch_queue.put(preprocess(images))

	def preprocess(data):
		'''Preprocess
	  	Args: 
	    	data: RGB batch (N * H * W * 3)
	  	Return:
	    	data_l: L channel batch (N * H * W * 1)
	    	gt_ab_313: ab discrete channel batch (N * H/4 * W/4 * 313)
	    	prior_boost_nongray: (N * H/4 * W/4 * 1) 
	  	'''
		warnings.filterwarnings("ignore")
	  	N = data.shape[0]
	  	H = data.shape[1]
	  	W = data.shape[2]
	  	data_l = data[:,:,:,0]
	  	data_u = data[:,:,:,1]
	  	data_v = data[:,:,:,2]
	  	return data_l, data_u, data_v


	def rgb_to_cieluv(rgb_image):


		# put in check for image size
		# image should be 224*224*3



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

	def batch(self):
	    """get batch
	    Returns:
	      images: 4-D ndarray [batch_size, height, width, 3]
	    """
	    #print(self.record_queue.qsize(), self.image_queue.qsize(), self.batch_queue.qsize())
		return self.batch_queue.get()
