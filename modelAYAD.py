from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *
from MIAS import *


def conv_out_size_same(size, stride):
	return int(math.ceil(float(size) / float(stride)))
def sigmoid_cross_entropy_with_logits(x, y):
		try:
				return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
		except:
				return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)
def noise(noise_type, noise_size, sample_num , noise_width=1.0):
		if(noise_type == "normal"):
				return np.random.normal(0, 2*noise_width, [sample_num, noise_size])
		else:
				return np.random.uniform(0, noise_width, [sample_num, noise_size])


#GANoptions is a dict containing all the options we use for our setup
#RIght now it has options for loss function and noise function
#noise type	normal/uniform
#noise size
#loss func	sigmoid,ls
#...

class DCGAN(object):
	def __init__(self, sess, crop=True,
				 batch_size=64, sample_num=64, output_height=64, output_width=64,
				z_dim=100, gf_dim=64, df_dim=64,	gfc_dim=1024, dfc_dim=1024, dataset_name='default',
				 input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None, input_class=None, GANoptions = None):
		"""

		Args:
		  sess: TensorFlow session
		  batch_size: The size of batch. Should be specified before training.
		  y_dim: (optional) Dimension of dim for y. [None]
		  z_dim: (optional) Dimension of dim for Z. [100]
		  gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
		  df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
		  gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
		  dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
		  c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
		"""

		if(input_class is not None):
			self.input_class=input_class
		if(GANoptions is not None):
				self.GANoptions=GANoptions

		self.sess = sess
		self.crop = crop

		self.batch_size = batch_size
		self.sample_num = sample_num

		self.input_width = input_class._images.shape[1]
		self.input_height = input_class._images.shape[2]
		self.output_height = output_height
		self.output_width = output_width

		self.y_dim = self.input_class.label_dim()


		if(self.GANoptions.get('noise size') is not None):
			self.z_dim = self.GANoptions.get('noise size')
		else:
			self.z_dim = z_dim

		self.gf_dim = gf_dim
		self.df_dim = df_dim

		self.gfc_dim = gfc_dim
		self.dfc_dim = dfc_dim

		# batch normalization : deals with poor initialization helps gradient flow
		self.d_bn1 = batch_norm(name='d_bn1')
		self.d_bn2 = batch_norm(name='d_bn2')

		if not self.y_dim:
			self.d_bn3 = batch_norm(name='d_bn3')

		self.g_bn0 = batch_norm(name='g_bn0')
		self.g_bn1 = batch_norm(name='g_bn1')
		self.g_bn2 = batch_norm(name='g_bn2')

		if not self.y_dim:
			self.g_bn3 = batch_norm(name='g_bn3')

		self.dataset_name =self.input_class.getName()
		self.checkpoint_dir=checkpoint_dir
		self.input_fname_pattern = input_fname_pattern

		if(input_class):	#AYAD
			self.c_dim = self.input_class.num_channels

		self.grayscale = (self.c_dim == 1)

		print("grayscale ", self.grayscale)
		print('dimensions', self.c_dim)
		print(self.input_class._images.shape)

		self.build_model()

	def build_model(self):
		if self.y_dim:
			self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

		if self.crop:
			image_dims = [self.output_height, self.output_width, self.c_dim]
		else:
			image_dims = [self.input_height, self.input_width, self.c_dim]

		self.inputs = tf.placeholder(
				tf.float32, [self.batch_size] + image_dims, name='real_images')
		self.sample_inputs = tf.placeholder(
				tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

		inputs = self.inputs
		sample_inputs = self.sample_inputs

		self.z = tf.placeholder(
			tf.float32, [None, self.z_dim], name='z')
		self.z_sum = histogram_summary("z", self.z)

		if self.y_dim:
			self.G = self.generator(self.z, self.y)
			self.D, self.D_logits = \
				self.discriminator(inputs, self.y, reuse=False)

			self.sampler = self.sampler(self.z, self.y)
			self.D_, self.D_logits_ = \
				self.discriminator(self.G, self.y, reuse=True)
		else:
			self.G = self.generator(self.z)
			self.D, self.D_logits, self.fmReal, self.fmReal2 = self.discriminator(inputs)	#ayad.adding feature matching

			self.sampler = self.sampler(self.z)
			self.D_, self.D_logits_, self.fmGen, self.fmGen2 = self.discriminator(self.G, reuse=True)

		self.d_sum = histogram_summary("d", self.D)
		self.d__sum = histogram_summary("d_", self.D_)
		self.G_sum = image_summary("G", self.G)

		self.featureMatchingLoss = self.featureMatching(self.fmReal,self.fmGen) + self.featureMatching(self.fmReal2,self.fmGen2)
		self.featureMatchingLoss *=2
		self.loss()#AYAD

		self.g_feature_matching_sum= scalar_summary("feature_matching_loss", self.featureMatchingLoss)
		self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
		self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

		self.d_loss = self.d_loss_real + self.d_loss_fake
		self.g_loss = self.g_loss +  self.featureMatchingLoss

		self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
		self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

		t_vars = tf.trainable_variables()

		self.d_vars = [var for var in t_vars if 'd_' in var.name]
		self.g_vars = [var for var in t_vars if 'g_' in var.name]

		self.saver = tf.train.Saver()

		# TENSORBOARD: display gradients
		self.fmloss = scalar_summary("Feature matching loss", self.featureMatchingLoss)

		self.gradientg1 = tf.gradients([self.g_loss],
																	 tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator/g_h1")[0])

		# self.gradient = []
		self.gradient_sumg1 = histogram_summary("h1gen", self.gradientg1)

		self.gradientg3 = tf.gradients([self.g_loss],
																	 tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator/g_h4")[0])
		# self.gradient = []
		self.gradient_sumg3 = histogram_summary("h4gen", self.gradientg3)

		self.gradientd3 = tf.gradients([self.d_loss],
																	 tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator/d_h3_lin")[
																			 0])
		# self.gradient = []
		self.gradient_sumd3 = histogram_summary("h3dis", self.gradientd3)

		self.gradientg1A = tf.reduce_mean(tf.abs(self.gradientg1))
		self.gradient_sumg1A = scalar_summary("h1AvgGrad", self.gradientg1A)

		self.gradientg3A = tf.reduce_mean(tf.abs(self.gradientg3))
		self.gradient_sumg3A = scalar_summary("h4AvgGrad", self.gradientg3A)

		self.gradientd3A = tf.reduce_mean(tf.abs(self.gradientd3))
		self.gradient_sumd3A = scalar_summary("d3AvgGrad", self.gradientd3A)

		self.activation = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator/activationg1")[0]
		self.activation_sum = histogram_summary("h1activ", self.activation)

		self.activationg4 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator/activationg4")[0]
		self.activation_sumg4 = histogram_summary("h3activ", self.activationg4)

		self.activationd3 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator/activationd3")[0]
		self.activation_sumd3 = histogram_summary("hd3activ", self.activationd3)



	def train(self, config):
		d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
			.minimize(self.d_loss, var_list=self.d_vars)
		g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
			.minimize(self.g_loss, var_list=self.g_vars)

		try:
			tf.global_variables_initializer().run()
		except:
			tf.initialize_all_variables().run()

		self.g_sum = merge_summary([self.z_sum, self.d__sum,
									self.G_sum, self.d_loss_fake_sum, self.g_loss_sum	, self.gradient_sumg1,
									self.gradient_sumg3, self.gradient_sumg1A,
									self.activation_sum, self.activation_sumg4, self.gradient_sumg3A, self.fmloss])
		self.d_sum = merge_summary(
			[self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum, self.gradient_sumd3, self.gradient_sumd3A,
			 self.activation_sumd3])

		self.writer = SummaryWriter("./logs", self.sess.graph)

		sample_z = noise(self.GANoptions['noise type'],self.z_dim, self.sample_num)

		if config.dataset == 'mnist':
			sample_inputs = self.data_X[0:self.sample_num]
			sample_labels = self.data_y[0:self.sample_num]
		else:
			sample_inputs, _ = self.input_class.next_batch(self.sample_num)

		counter = 1
		start_time = time.time()
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			counter = checkpoint_counter
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
		print("going into loop")
		self.data = self.input_class._images
		counter=1
		for epoch in xrange(config.epoch):
			print("in epoch")

			if config.dataset == 'mnist':
				batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size
			else:
				batch_idxs = min(self.data.shape[0], config.train_size) // config.batch_size

			for idx in xrange(0, batch_idxs):
				if config.dataset == 'mnist':
					batch_images = self.data_X[idx * config.batch_size:(idx + 1) * config.batch_size]
					batch_labels = self.data_y[idx * config.batch_size:(idx + 1) * config.batch_size]
				else:
					batch_images, _= self.input_class.next_batch(self.batch_size)  #AYAD

				batch_z = noise(self.GANoptions['noise type'],self.z_dim, self.sample_num) \
						.astype(np.float32)

				if config.dataset == 'mnist':
					# Update D network
					_, summary_str = self.sess.run([d_optim, self.d_sum],
												   feed_dict={
													   self.inputs: batch_images,
													   self.z: batch_z,
													   self.y: batch_labels,
												   })
					self.writer.add_summary(summary_str, counter)

					# Update G network
					_, summary_str = self.sess.run([g_optim, self.g_sum],
												   feed_dict={
													   self.z: batch_z,
													   self.y: batch_labels,
												   })
					self.writer.add_summary(summary_str, counter)

					# Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
					_, summary_str = self.sess.run([g_optim, self.g_sum],
												   feed_dict={self.z: batch_z, self.y: batch_labels})
					self.writer.add_summary(summary_str, counter)

					errD_fake = self.d_loss_fake.eval({
						self.z: batch_z,
						self.y: batch_labels
					})
					errD_real = self.d_loss_real.eval({
						self.inputs: batch_images,
						self.y: batch_labels
					})
					errG = self.g_loss.eval({
						self.z: batch_z,
						self.y: batch_labels
					})
				else:
					# Update D network
					_, summary_str = self.sess.run([d_optim, self.d_sum],
												   feed_dict={self.inputs: batch_images, self.z: batch_z})
					self.writer.add_summary(summary_str, counter)

					######################################################################
					# Update D network
					_, summary_str = self.sess.run([d_optim, self.d_sum],
												   feed_dict={self.inputs: batch_images, self.z: batch_z})
					self.writer.add_summary(summary_str, counter)
					# Update D network
					_, summary_str = self.sess.run([d_optim, self.d_sum],
												   feed_dict={self.inputs: batch_images, self.z: batch_z})
					self.writer.add_summary(summary_str, counter)
					# Update D network
					_, summary_str = self.sess.run([d_optim, self.d_sum],
												   feed_dict={self.inputs: batch_images, self.z: batch_z})
					self.writer.add_summary(summary_str, counter)
					# Update D network
					_, summary_str = self.sess.run([d_optim, self.d_sum],
												   feed_dict={self.inputs: batch_images, self.z: batch_z})
					self.writer.add_summary(summary_str, counter)
					# Update D network
					_, summary_str = self.sess.run([d_optim, self.d_sum],
												   feed_dict={self.inputs: batch_images, self.z: batch_z})
					self.writer.add_summary(summary_str, counter)
					# Update D network
					_, summary_str = self.sess.run([d_optim, self.d_sum],
												   feed_dict={self.inputs: batch_images, self.z: batch_z})
					self.writer.add_summary(summary_str, counter)
					######################################################################

					# Update G network
					_, summary_str = self.sess.run([g_optim, self.g_sum],
												   feed_dict={self.inputs: batch_images, self.z: batch_z})
					self.writer.add_summary(summary_str, counter)


					# Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
					_, summary_str = self.sess.run([g_optim, self.g_sum],
												   feed_dict={ self.inputs: batch_images, self.z: batch_z})
					self.writer.add_summary(summary_str, counter)
					######################AYAD#################################
					#More generator runs
					# Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
				# 	_, summary_str = self.sess.run([g_optim, self.g_sum],
				# 								   feed_dict={ self.inputs: batch_images, self.z: batch_z})
				# 	self.writer.add_summary(summary_str, counter)
				# 	######################################################

					errD_fake = self.d_loss_fake.eval({self.z: batch_z})
					errD_real = self.d_loss_real.eval({self.inputs: batch_images})
					errG = self.g_loss.eval({self.inputs: batch_images,self.z: batch_z})

				counter += 1
				print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
					  % (epoch, idx, batch_idxs,
						 time.time() - start_time, errD_fake + errD_real, errG))


				if np.mod(counter, 100) == 1:
					if config.dataset == 'mnist':
						samples, d_loss, g_loss = self.sess.run(
							[self.sampler, self.d_loss, self.g_loss],
							feed_dict={
								self.z: sample_z,
								self.inputs: sample_inputs,
								self.y: sample_labels,
							}
						)
						manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
						manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
						save_images(samples, [manifold_h, manifold_w],
									'./{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
						print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
					else:
						try:
							samples, d_loss, g_loss = self.sess.run(
								[self.sampler, self.d_loss, self.g_loss],
								feed_dict={
									self.z: sample_z,
									self.inputs: sample_inputs,
								},
							)
							manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
							manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
							print(samples.shape,epoch,idx,"999999999999999999")
							save_images(samples, [manifold_h, manifold_w],
										'./{}/train_{:02d}_{:04d}.png'.format("samples", epoch, idx))
							########################################################################
						# 	save_images(batch_images, [manifold_h, manifold_w],
						# 							'./{}/data_{:02d}_{:04d}.png'.format("samples", epoch, idx))
							########################################################################
							print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
							#print("[Sample] dicrim real: %.8f, discrim fake: %.8f" % (self.D_logits, self.D_logits_))
						except:
							print("one pic error!...")

				if np.mod(counter, 200) == 2:
					self.save(config.checkpoint_dir, counter)

	#Ayad last param should sepcify wich layer to return for feature matching(TODO)
	def discriminator(self, image, y=None, reuse=False, FM =2 ):
			with tf.variable_scope("discriminator") as scope:
					if reuse:
							scope.reuse_variables()

					if not self.y_dim:
							h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
							h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
							h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
							h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))

							h3Activ = tf.get_variable('activationd3', h3.shape,
																				initializer=tf.random_normal_initializer(stddev=0.02))
							h3Activ = h3

							h3 = tf.contrib.layers.flatten(h3)
							h3 = minibatch_discrimination(h3, self.z_dim)
							# h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')
							h4 = linear(tf.contrib.layers.flatten(h3), 1, 'd_h3_lin')

							#h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
							return tf.nn.sigmoid(h4), h4, h2 , h3  # ayad. added the third return for feature matching
					else:
							yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
							x = conv_cond_concat(image, yb)

							h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
							h0 = conv_cond_concat(h0, yb)

							h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
							h1 = tf.reshape(h1, [self.batch_size, -1])
							h1 = concat([h1, y], 1)

							h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, scope='d_h2_lin')))
							h2 = concat([h2, y], 1)

							h3 = linear(h2, 1, 'd_h3_lin')

							return tf.nn.sigmoid(h3), h3 , h2 # ayad

	def discriminator2(self, image, y=None, reuse=False, FM =2 ):
			with tf.variable_scope("discriminator") as scope:
					if reuse:
							scope.reuse_variables()

					if not self.y_dim:
							h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
							h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
							h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
							h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))

						# 	h3Activ = tf.get_variable('activationd3', h3.shape,
						# 														initializer=tf.random_normal_initializer(stddev=0.02))
						# 	h3Activ = h3

							h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
							return tf.nn.sigmoid(h4), h4, h2   # ayad. added the third return for feature matching
					else:
							yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
							x = conv_cond_concat(image, yb)

							h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
							h0 = conv_cond_concat(h0, yb)

							h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
							h1 = tf.reshape(h1, [self.batch_size, -1])
							h1 = concat([h1, y], 1)

							h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, scope='d_h2_lin')))
							h2 = concat([h2, y], 1)

							h3 = linear(h2, 1, 'd_h3_lin')

							return tf.nn.sigmoid(h3), h3  # ayad

	def generator(self, z, y=None):
			with tf.variable_scope("generator") as scope:
					if not self.y_dim:
							s_h, s_w = self.output_height, self.output_width
							s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
							s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
							s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
							s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

							# project `z` and reshape
							self.z_, self.h0_w, self.h0_b = linear(
									z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True)

							self.h0 = tf.reshape(
									self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
							h0 = tf.nn.relu(self.g_bn0(self.h0))
							#################################layer 1######################################
							self.h1, self.h1_w, self.h1_b = deconv2d(
									h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1', with_w=True)
							h1 = tf.nn.relu(self.g_bn1(self.h1))

							h1Activ = tf.get_variable('activationg1', h1.shape,
																				initializer=tf.random_normal_initializer(stddev=0.02))
							h1Activ = h1
							###########################################################################
							h2, self.h2_w, self.h2_b = deconv2d(
									h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)
							h2 = tf.nn.relu(self.g_bn2(h2))
							#
							h3, self.h3_w, self.h3_b = deconv2d(
									h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)
							h3 = tf.nn.relu(self.g_bn3(h3))

							###############################layer 4################################
							h4, self.h4_w, self.h4_b = deconv2d(
									h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

							h4Activ = tf.get_variable('activationg4', h4.shape,
																				initializer=tf.random_normal_initializer(stddev=0.02))
							h4Activ = h4
							###############################################################################
							return tf.nn.tanh(h4)
					else:
							s_h, s_w = self.output_height, self.output_width
							s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
							s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

							# yb = tf.expand_dims(tf.expand_dims(y, 1),2)
							yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
							z = concat([z, y], 1)

							h0 = tf.nn.relu(
									self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
							h0 = concat([h0, y], 1)

							h1 = tf.nn.relu(self.g_bn1(
									linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin')))
							h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

							h1 = conv_cond_concat(h1, yb)

							h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
																									[self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
							h2 = conv_cond_concat(h2, yb)

							return tf.nn.sigmoid(
									deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

	def sampler(self, z, y=None):
			with tf.variable_scope("generator") as scope:
					scope.reuse_variables()

					if not self.y_dim:
							s_h, s_w = self.output_height, self.output_width
							s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
							s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
							s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
							s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

							# project `z` and reshape
							h0 = tf.reshape(
									linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin'),
									[-1, s_h16, s_w16, self.gf_dim * 8])
							h0 = tf.nn.relu(self.g_bn0(h0, train=False))
							######################################################################
							h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1')
							h1 = tf.nn.relu(self.g_bn1(h1, train=False))
							h1Activ = tf.get_variable('activationg1', h1.shape,
																				initializer=tf.random_normal_initializer(stddev=0.02))
							h1Activ = h1
							#####################################################################
							h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2')
							h2 = tf.nn.relu(self.g_bn2(h2, train=False))

							h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3')
							h3 = tf.nn.relu(self.g_bn3(h3, train=False))
							##############################################################################
							h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')
							#################################################################################
							return tf.nn.tanh(h4)
					else:
							s_h, s_w = self.output_height, self.output_width
							s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
							s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

							# yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
							yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
							z = concat([z, y], 1)

							h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
							h0 = concat([h0, y], 1)

							h1 = tf.nn.relu(self.g_bn1(
									linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin'), train=False))
							h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
							h1 = conv_cond_concat(h1, yb)

							h2 = tf.nn.relu(self.g_bn2(
									deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
							h2 = conv_cond_concat(h2, yb)

							return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

	def sampler2(self, z, y=None):
			with tf.variable_scope("generator") as scope:
					scope.reuse_variables()

					if not self.y_dim:
							s_h, s_w = self.output_height, self.output_width
							s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
							s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
							s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
							s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

							# project `z` and reshape
							h0 = tf.reshape(
									linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin'),
									[-1, s_h16, s_w16, self.gf_dim * 8])
							h0 = tf.nn.relu(self.g_bn0(h0, train=False))
							######################################################################
							h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1')
							h1 = tf.nn.relu(self.g_bn1(h1, train=False))
						# 	h1Activ = tf.get_variable('activationg1', h1.shape,
						# 														initializer=tf.random_normal_initializer(stddev=0.02))
						# 	h1Activ = h1
							#####################################################################
							h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2')
							h2 = tf.nn.relu(self.g_bn2(h2, train=False))

							h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3')
							h3 = tf.nn.relu(self.g_bn3(h3, train=False))
							##############################################################################
							h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')
							#################################################################################
							return tf.nn.tanh(h4)
					else:
							s_h, s_w = self.output_height, self.output_width
							s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
							s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

							# yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
							yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
							z = concat([z, y], 1)

							h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
							h0 = concat([h0, y], 1)

							h1 = tf.nn.relu(self.g_bn1(
									linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin'), train=False))
							h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
							h1 = conv_cond_concat(h1, yb)

							h2 = tf.nn.relu(self.g_bn2(
									deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
							h2 = conv_cond_concat(h2, yb)

							return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

	def loss(self):
			#sperated the loss function part so we can later modify it easily
			label_smooth=self.GANoptions['label smooth']
			label_smooth = True
			if(label_smooth is True):
					d_real_labels = tf.truncated_normal(self.D.get_shape(), mean=0.88, stddev= 0.1)
					g_fake_labels = tf.truncated_normal(self.D.get_shape(), mean=0.88, stddev= 0.1)
			else:
					d_real_labels = tf.ones_like(self.D)
					g_fake_labels = tf.ones_like(self.D_)

			self.d_loss_real = tf.reduce_mean(
					sigmoid_cross_entropy_with_logits(self.D_logits, d_real_labels))
			self.d_loss_fake = tf.reduce_mean(
					sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
			self.g_loss = tf.reduce_mean(
					sigmoid_cross_entropy_with_logits(self.D_logits_, g_fake_labels))

			##############Least squares#########################################
			self.d_loss_real_LS = tf.reduce_mean((self.D_logits -  d_real_labels)**2 )
			self.d_loss_fake_LS = tf.reduce_mean((self.D_logits_ -  tf.zeros_like(self.D_))**2 )
			self.g_loss_LS = tf.reduce_mean( (self.D_logits_ -  g_fake_labels)**2 )
			####################################################################

			self.d_loss_real += self.d_loss_real_LS
			self.d_loss_fake += self.d_loss_fake_LS
			self.g_loss += self.g_loss_LS


	def load_mnist(self):
		data_dir = os.path.join(".\data", self.dataset_name)

		# fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
		fd = open("C:/Users/Ayad/PycharmProjects/DCGANS/DCGAN-tensorflow-master/data/mnist/train-images.idx3-ubyte")
		loaded = np.fromfile(file=fd, dtype=np.uint8)
		trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

		fd = open(os.path.join("C:/Users/Ayad/PycharmProjects/DCGANS/DCGAN-tensorflow-master/data/mnist/",
							   'train-labels.idx1-ubyte'))
		loaded = np.fromfile(file=fd, dtype=np.uint8)
		trY = loaded[8:].reshape((60000)).astype(np.float)

		fd = open(os.path.join(data_dir, 't10k-images.idx3-ubyte'))
		loaded = np.fromfile(file=fd, dtype=np.uint8)
		teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

		fd = open(os.path.join(data_dir, 't10k-labels.idx1-ubyte'))
		loaded = np.fromfile(file=fd, dtype=np.uint8)
		teY = loaded[8:].reshape((10000)).astype(np.float)

		trY = np.asarray(trY)
		teY = np.asarray(teY)

		X = np.concatenate((trX, teX), axis=0)
		y = np.concatenate((trY, teY), axis=0).astype(np.int)

		seed = 547
		np.random.seed(seed)
		np.random.shuffle(X)
		np.random.seed(seed)
		np.random.shuffle(y)

		y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
		for i, label in enumerate(y):
			y_vec[i, y[i]] = 1.0

		return X / 255., y_vec

	@property
	def model_dir(self):
		return "{}_{}_{}_{}".format(
			self.dataset_name, self.batch_size,
			self.output_height, self.output_width)

	def save(self, checkpoint_dir, step):
		model_name = "DCGAN.model"
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
						os.path.join(checkpoint_dir, model_name),
						global_step=step)

	def load(self, checkpoint_dir):
		import re
		print(" [*] Reading checkpoints...")
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
		########################checkpoint#########################################
		#checkpoint_dir = os.path.join(checkpoint_dir, "croppingseqcropskip002cropsize128128cropstep8_32_128_128")
		print("**************",checkpoint_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
			print(" [*] Success to read {}".format(ckpt_name))
			return True, counter
		else:
			print(" [*] Failed to find a checkpoint")
			return False, 0

	#This function simply returns an (l1 norm / numofcomponents) between a discriminator activation tensor
	#on real data, and one from generator data
	#In training we use two full tensors.
	#This function is also used in the anomaly detection, but we use the mean discriminator
	#activation, and the activation of the query image
	def featureMatching(self, real, fake):
		return tf.reduce_mean( tf.abs(real-fake) )
