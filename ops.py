import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

try:
		image_summary = tf.image_summary
		scalar_summary = tf.scalar_summary
		histogram_summary = tf.histogram_summary
		merge_summary = tf.merge_summary
		SummaryWriter = tf.train.SummaryWriter
except:
		image_summary = tf.summary.image
		scalar_summary = tf.summary.scalar
		histogram_summary = tf.summary.histogram
		merge_summary = tf.summary.merge
		SummaryWriter = tf.summary.FileWriter

if "concat_v2" in dir(tf):
		def concat(tensors, axis, *args, **kwargs):
				return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
		def concat(tensors, axis, *args, **kwargs):
				return tf.concat(tensors, axis, *args, **kwargs)


class batch_norm(object):
		def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
				with tf.variable_scope(name):
						self.epsilon = epsilon
						self.momentum = momentum
						self.name = name

		def __call__(self, x, train=True):
				return tf.contrib.layers.batch_norm(x,
																						decay=self.momentum,
																						updates_collections=None,
																						epsilon=self.epsilon,
																						scale=True,
																						is_training=train,
																						scope=self.name)


def conv_cond_concat(x, y):
		"""Concatenate conditioning vector on feature map axis."""
		x_shapes = x.get_shape()
		y_shapes = y.get_shape()
		return concat([
				x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def conv2d(input_, output_dim,
					 k_h=4, k_w=4, d_h=2, d_w=2, stddev=0.02,
					 name="conv2d"):
	print("input ",input_.get_shape(),name )
	with tf.variable_scope(name):
		w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
												initializer=tf.orthogonal_initializer())#initializer=tf.truncated_normal_initializer(stddev=stddev))
		conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

		biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
		conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
		print(conv.get_shape(), name)

	return conv


def deconv2d(input_, output_shape,
						 k_h=4, k_w=4, d_h=2, d_w=2, stddev=0.02,
						 name="deconv2d", with_w=False):
		with tf.variable_scope(name):
				# filter : [height, width, output_channels, in_channels]
				w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
														initializer=tf.orthogonal_initializer())#initializer=tf.random_normal_initializer(stddev=stddev))

				try:
						deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
																						strides=[1, d_h, d_w, 1])

				# Support for verisons of TensorFlow before 0.7.0
				except AttributeError:
						deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
																		strides=[1, d_h, d_w, 1])

				biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
				deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

				print(deconv.get_shape(), name)

				if with_w:
						return deconv, w, biases
				else:
						return deconv


# output_shape [height , width ]
def upsample_conv(input_, output_shape,
									name="upsample", k_h=8, k_w=8, d_h=1, d_w=1, stddev=0.02 ,with_w=False):


		# input_=tf.image.resize_images(input_, output_shape[1:3],  method="NEAREST_NEIGHBOR")
		input_ = tf.image.resize_nearest_neighbor (input_, output_shape[1:3])

		with tf.variable_scope(name):
				w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_shape[3]],
														initializer=tf.truncated_normal_initializer(stddev=stddev))
				conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

				biases = tf.get_variable('biases', [output_shape[3]], initializer=tf.constant_initializer(0.0))
				conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
				print(conv.get_shape(), name)

		if with_w:
			return conv ,4,4
		else:
			return conv



def check_shape(h_size, im_size, stride):
	if h_size != (im_size + stride - 1) // stride:
			print ("Need h_size == (im_size + stride - 1) // stride")
			print ("h_size: ", h_size)
			print ("im_size: ", im_size)
			print ("stride: ", stride)
			print ("(im_size + stride - 1) / float(stride): ", (im_size + stride - 1) / float(stride))
			raise ValueError()

def special_deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=3, d_w=3, stddev=0.02,
             name="deconv2d", with_w=False,
             init_bias=0.):
	# designed to reduce padding and stride artifacts in the generator

	# If the following fail, it is hard to avoid grid pattern artifacts
	assert k_h % d_h == 0
	assert k_w % d_w == 0

	with tf.variable_scope(name):
	# filter : [height, width, output_channels, in_channels]
		w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))



		check_shape(int(input_.get_shape()[1]), output_shape[1] + k_h, d_h)
		check_shape(int(input_.get_shape()[2]), output_shape[2] + k_w, d_w)

		deconv = tf.nn.conv2d_transpose(input_, w, output_shape=[output_shape[0],
			output_shape[1] + k_h, output_shape[2] + k_w, output_shape[3]],
                                strides=[1, d_h, d_w, 1])
		deconv = tf.slice(deconv, [0, k_h // 2, k_w // 2, 0], output_shape)

		biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(init_bias))
		deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

		if (with_w):
			return deconv, w, biases
		else:
			return deconv





def lrelu(x, leak=0.2, name="lrelu"):
		return tf.maximum(x, leak * x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
		shape = input_.get_shape().as_list()

		with tf.variable_scope(scope or "Linear"):
				matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
																 tf.random_normal_initializer(stddev=stddev))
				bias = tf.get_variable("bias", [output_size],
															 initializer=tf.constant_initializer(bias_start))
				if with_w:
						return tf.matmul(input_, matrix) + bias, matrix, bias
				else:
						return tf.matmul(input_, matrix) + bias


def linear2(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
		shape = input_.get_shape().as_list()

		with tf.variable_scope("other"):
				print(shape, scope)
				matrix = tf.get_variable("Matrix" + scope, [shape[1], output_size], tf.float32,
																 tf.random_normal_initializer(stddev=stddev))
				bias = tf.get_variable("bias", [output_size],
															 initializer=tf.constant_initializer(bias_start))
				if with_w:
						return tf.matmul(input_, matrix) + bias, matrix, bias
				else:
						return tf.matmul(input_, matrix) + bias

def minibatch_discrimination(input_layer,num_kernels, dim_per_kernel =5, name='minibatch_discrim'):

    # batch_size = input_layer.shape[0]
    # num_features = input_layer.shape[1]
    batch_size = input_layer.get_shape().as_list()[0]
    num_features = input_layer.get_shape().as_list()[1]

    W = tf.get_variable('W', [num_features, num_kernels*dim_per_kernel],
                      initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable('b', [num_kernels], initializer=tf.constant_initializer(0.0))
    activation = tf.matmul(input_layer, W)

    activation = tf.reshape(activation, [batch_size, num_kernels, dim_per_kernel])
    tmp1 = tf.expand_dims(activation, 3)
    tmp2 = tf.transpose(activation, perm=[1,2,0])
    tmp2 = tf.expand_dims(tmp2, 0)
    abs_diff = tf.reduce_sum(tf.abs(tmp1 - tmp2), reduction_indices=[2])
    f = tf.reduce_sum(tf.exp(-abs_diff), reduction_indices=[2])
    f = f + b
    return f
