import numpy
from PIL import Image

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
			luv_image[i][j][0] = l
			luv_image[i][j][1] = u
			luv_image[i][j][2] = v

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

img = Image.open('test.jpg')
rgb_matrix = numpy.array(img)
luv_matrix = rgb_to_cieluv(rgb_matrix)
print(luv_matrix)