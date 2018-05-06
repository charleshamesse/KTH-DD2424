import math
from glob import glob
import os
import time
import tensorflow as tf
import numpy as np
from ops import lrelu, conv2d, conv_cond_concat, linear, concat, deconv2d, batch_norm
from utils import imread, sigmoid_cross_entropy_with_logits, get_image, save_images, image_manifold_size


class DCGAN(object):
    def __init__(self, sess):
        """MODEL PARAMS. INIT FROM INPUT LATER"""
        self.image_shape = 108, 108
        self.z_dim = 100
        self.dataset_name = "reptiles"
        self.checkpoint_dir = "checkpoints"
        self.model_dir = 'models'
        self.data_dir = "../../datadir"  # TODO fix
        self.batch_size = 64
        self.gf_dim = 64  # Dimension of gen filters in first conv layer. [64]
        self.df_dim = 64  # Dimension of discrim filters in first conv layer. [64]
        self.sess = sess
        self.sample_num = 64
        self.learning_rate = 0.0002
        self.beta1 = 0.5  # Momentum term of adam [0.5]
        self.epochs = 2

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')  # TODO maybe remove this line
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')  # TODO maybe remove this line

        self.data = glob(os.path.join(self.data_dir, self.dataset_name, '*.jpg'))
        imread_img = imread(self.data[0])
        if len(imread_img.shape) >= 3:  # check if image is a non-grayscale image by checking channel number
            self.c_dim = imread(self.data[0]).shape[-1]
        else:
            self.c_dim = 1
        self.grayscale = (self.c_dim == 1)
        self.build_model()

    @staticmethod
    def conv_out_size_same(size, stride):
        return int(math.ceil(float(size) / float(stride)))

    def build_model(self):
        image_dims = [self.image_shape[0], self.image_shape[1], self.c_dim]

        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')

        inputs = self.inputs

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        self.G = self.generator(self.z)
        self.D, self.D_logits = self.discriminator(inputs, reuse=False)
        # self.sampler = self.sampler(self.z, self.y)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))

        self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))

        self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        self.d_loss = self.d_loss_real + self.d_loss_fake

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

            return tf.nn.sigmoid(h4), h4

    def generator(self, z):
        with tf.variable_scope("generator") as scope:
            s_h, s_w = self.image_shape
            s_h2, s_w2 = self.conv_out_size_same(s_h, 2), self.conv_out_size_same(s_w, 2)
            s_h4, s_w4 = self.conv_out_size_same(s_h2, 2), self.conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = self.conv_out_size_same(s_h4, 2), self.conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = self.conv_out_size_same(s_h8, 2), self.conv_out_size_same(s_w8, 2)

            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, s_h8, s_w8,
                                                          self.gf_dim * 4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2],
                                                name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(
                h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(
                h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)

    def train(self):
        d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(self.g_loss, var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        # sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

        # sample_files = self.data[0:self.sample_num]
        # sample = [get_image(sample_file,
        #           input_height=self.image_shape[0],
        #           input_width=self.image_shape[1],
        #           resize_height=self.image_shape[0],
        #           resize_width=self.image_shape[1]) for sample_file in sample_files]

        # sample_inputs = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in range(self.epochs):
            self.data = glob(os.path.join(self.data_dir, self.dataset_name, "*.jpg"))
            batch_idxs = len(self.data) // self.batch_size

            for idx in range(0, batch_idxs):
                batch_files = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch = [get_image(batch_file,
                         input_height=self.image_shape[0],
                         input_width=self.image_shape[1],
                         resize_height=self.image_shape[0],
                         resize_width=self.image_shape[1]) for batch_file in batch_files]

                batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]) \
                    .astype(np.float32)

                # Update D network
                self.sess.run([d_optim], feed_dict={self.inputs: batch_images, self.z: batch_z})

                self.sess.run([g_optim], feed_dict={self.z: batch_z})

                # TODO Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                self.sess.run([g_optim], feed_dict={self.z: batch_z})

                errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                errD_real = self.d_loss_real.eval({self.inputs: batch_images})
                errG = self.g_loss.eval({self.z: batch_z})

                counter += 1
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"\
                      % (epoch, self.epochs, idx, batch_idxs,
                         time.time() - start_time, errD_fake + errD_real, errG))

                # if np.mod(counter, 100) == 1:
                #     try:
                #         samples, d_loss, g_loss = self.sess.run(
                #             [self.sampler, self.d_loss, self.g_loss],
                #             feed_dict={
                #                 self.z: sample_z,
                #                 self.inputs: sample_inputs,
                #             },
                #         )
                #         save_images(samples, image_manifold_size(samples.shape[0]),
                #                     './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                #         print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                #     except:
                #         print("one pic error!...")

                if np.mod(counter, 500) == 2:
                    self.save(self.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        # TODO make sure we dont overwrite existing models.
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

