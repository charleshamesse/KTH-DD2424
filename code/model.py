import math
from glob import glob
import os
import json
import time
import tensorflow as tf
from keras.datasets import cifar10
import numpy as np
import config
from ops import lrelu, conv2d, conv_cond_concat, linear, concat, deconv2d, batch_norm
from utils import imread, sigmoid_cross_entropy_with_logits, get_image

class DCGAN(object):
    def __init__(self, sess):
        """MODEL PARAMS. INIT FROM INPUT LATER"""
        self.data_dir = config.DATA['data_dir']
        self.dataset_name = config.DATA['dataset_name']
        self.data_limit = config.MODEL['data_limit']

        if self.dataset_name == 'cifar10':
            self.image_shape = 32, 32
            (x_train, _), (_, _) = cifar10.load_data()
            self.data = x_train[0:self.data_limit] / 127.5 - 1 
            self.c_dim = 3
            self.grayscale = 0
        else:  # reptile set
            self.image_shape = 108, 108
            self.data = glob(os.path.join(self.data_dir, self.dataset_name, "*.jpg"))[0:self.data_limit]
            imread_img = imread(self.data[0])
            if len(imread_img.shape) >= 3:  # check if image is a non-grayscale image by checking channel number
                self.c_dim = imread(self.data[0]).shape[-1]
            else:
                self.c_dim = 1
            self.grayscale = (self.c_dim == 1)

        self.batch_size = config.MODEL['batch_size']
        self.batch_idxs = len(self.data) // self.batch_size
        self.sample_dir = config.DATA['sample_dir']
        self.checkpoint_dir = config.MODEL['checkpoint_dir']
        self.model_dir = config.MODEL['model_dir']
        self.z_dim = 100
        self.batch_size = config.MODEL['batch_size']
        self.gf_dim = 64  # Dimension of gen filters in first conv layer. [64]
        self.df_dim = 64  # Dimension of discrim filters in first conv layer. [64]
        self.sess = sess
        self.sample_num = 9
        self.learning_rate = 0.0002
        self.beta1 = 0.5  # Momentum term of adam [0.5]
        self.epochs = config.MODEL['epochs']
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')  # TODO maybe remove this line
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')  # TODO maybe remove this line

        self.use_spectral_norm = True
        self.build_model()

    @staticmethod
    def conv_out_size_same(size, stride):
        return int(math.ceil(float(size) / float(stride)))

    def build_model(self):
        # Placeholders
        image_dims = [self.image_shape[0], self.image_shape[1], self.c_dim]
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        # Evaluate networks
        self.G = self.generator(self.z)
        self.D_real, self.D_real_logits = self.discriminator(self.inputs, reuse=False, update_collection='NO_OPS')
        self.D_fake, self.D_fake_logits = self.discriminator(self.G, reuse=True, update_collection=None)
        
        # Losses
        self.D_real_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_real_logits, tf.ones_like(self.D_real)))
        self.D_fake_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_fake_logits, tf.zeros_like(self.D_fake)))
        self.D_loss = self.D_real_loss + self.D_fake_loss
        self.G_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_fake_logits, tf.ones_like(self.D_fake)))
        
        # Variables
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        # Saver
        self.saver = tf.train.Saver()

    def discriminator(self, image, reuse=False, update_collection=tf.GraphKeys.UPDATE_OPS):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            
            # Convolution blocks
            # If we use spectral norm, we don't use batch norm
            if self.use_spectral_norm: 
                h0 = lrelu(conv2d(image, self.df_dim,  name='d_h0_conv', spectral_normed=self.use_spectral_norm, update_collection=update_collection))
                h1 = lrelu(conv2d(h0, self.df_dim * 2, name='d_h1_conv', spectral_normed=self.use_spectral_norm, update_collection=update_collection))
                h2 = lrelu(conv2d(h1, self.df_dim * 4, name='d_h2_conv', spectral_normed=self.use_spectral_norm, update_collection=update_collection))
                h3 = lrelu(conv2d(h2, self.df_dim * 8, name='d_h3_conv', spectral_normed=self.use_spectral_norm, update_collection=update_collection))
                h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin', spectral_normed=self.use_spectral_norm, update_collection=update_collection)
            else:
                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv', spectral_normed=self.use_spectral_norm, update_collection=update_collection))
                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv', spectral_normed=self.use_spectral_norm, update_collection=update_collection)))
                h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv', spectral_normed=self.use_spectral_norm, update_collection=update_collection)))
                h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv', spectral_normed=self.use_spectral_norm, update_collection=update_collection)))
                h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin', spectral_normed=self.use_spectral_norm, update_collection=update_collection)

            return tf.nn.sigmoid(h4), h4

    def generator(self, z, train=True):
        with tf.variable_scope("generator") as scope:
            # train is True when training, False when sampling
            batch_size = self.batch_size
            if not train: # When sampling
                scope.reuse_variables()
                batch_size = self.sample_num

            # Convolution parameter sizes
            s_h, s_w = self.image_shape
            s_h2, s_w2 = self.conv_out_size_same(s_h, 2), self.conv_out_size_same(s_w, 2)
            s_h4, s_w4 = self.conv_out_size_same(s_h2, 2), self.conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = self.conv_out_size_same(s_h4, 2), self.conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = self.conv_out_size_same(s_h8, 2), self.conv_out_size_same(s_w8, 2)

            # Project z, reshape and go through 4 convolution blocks (deconv, batch norm, relu)
            self.z_ = linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin')
            self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0, train=train))
            h1 = tf.nn.relu(self.g_bn1(deconv2d(h0, [batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1'), train=train))
            h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2'), train=train))
            h3 = tf.nn.relu(self.g_bn3(deconv2d(h2, [batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3'), train=train))
            h4 = tf.nn.tanh(deconv2d(h3, [batch_size, s_h, s_w, self.c_dim], name='g_h4'))

            return h4