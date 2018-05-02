import tensorflow as tf
from utils import *
from network import Network
from skimage.io import imsave
from skimage.transform import resize
import cv2

img = cv2.imread('iitb_gray.png')
if len(img.shape) == 3:
  img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img = img[None, :, :, None]
data_l = (img.astype(dtype=np.float32)) / 255.0 * 100 - 50

#data_l = tf.placeholder(tf.float32, shape=(None, None, None, 1))
autocolor = Network(train=False)

final_u,final_v = autocolor.create_net(data_l)

saver = tf.train.Saver()
with tf.Session() as sess:
  saver.restore(sess, 'models/model.ckpt')
  conv8_313 = sess.run(conv8_313)

img_rgb = decode(data_l, final_u,final_v,2.63)
imsave('iitb_color.png', img_rgb)																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																													
