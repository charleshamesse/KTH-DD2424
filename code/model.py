import math
from glob import glob
import os
import json
import time
import tensorflow as tf
import numpy as np
import config
from ops import lrelu, conv2d, conv_cond_concat, linear, concat, deconv2d, batch_norm
from utils import imread, sigmoid_cross_entropy_with_logits, get_image, save_images, image_manifold_size, NumpyEncoder
from inception_score import get_inception_score

class DCGAN(object):
    def __init__(self, sess):
        """MODEL PARAMS. INIT FROM INPUT LATER"""
        self.image_shape = 108, 108
        self.z_dim = 100
        self.dataset_name = config.DATA['dataset_name']
        self.data_dir = config.DATA['data_dir']
        self.sample_dir = config.DATA['sample_dir']
        self.results_dir = config.DATA['results_dir']
        self.checkpoint_dir = config.MODEL['checkpoint_dir']
        self.model_dir = config.MODEL['model_dir']
        self.batch_size = 4
        self.gf_dim = 64  # Dimension of gen filters in first conv layer. [64]
        self.df_dim = 64  # Dimension of discrim filters in first conv layer. [64]
        self.sess = sess
        self.sample_num = 1000
        self.learning_rate = 0.0002
        self.beta1 = 0.5  # Momentum term of adam [0.5]
        self.epochs = 5
        self.results = {
            "d_loss": [],
            "g_loss": [],
            "is": []
        }
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')  # TODO maybe remove this line
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')  # TODO maybe remove this line

        self.use_spectral_norm = True

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
                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv', spectral_normed=self.use_spectral_norm, update_collection=update_collection))
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

    def generator(self, z):
        with tf.variable_scope("generator") as scope:
            # Convolution parameter sizes
            s_h, s_w = self.image_shape
            s_h2, s_w2 = self.conv_out_size_same(s_h, 2), self.conv_out_size_same(s_w, 2)
            s_h4, s_w4 = self.conv_out_size_same(s_h2, 2), self.conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = self.conv_out_size_same(s_h4, 2), self.conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = self.conv_out_size_same(s_h8, 2), self.conv_out_size_same(s_w8, 2)

            # Project z and reshape
            self.z_ = linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin')
            self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))
            
            # Deconvolution blocks (deconv, batch norm, relu)
            h1 = tf.nn.relu(self.g_bn1(deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1')))
            h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2')))
            h3 = tf.nn.relu(self.g_bn3(deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3')))
            h4 = tf.nn.tanh(deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4'))

            return h4

    def train(self):
        d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(self.D_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(self.G_loss, var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in range(self.epochs):
            self.data = glob(os.path.join(self.data_dir, self.dataset_name, "*.jpg"))[0:8]
            batch_idxs = len(self.data) // self.batch_size

            epoch_d_loss = 0
            epoch_g_loss = 0
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

                errD_fake = self.D_fake_loss.eval({self.z: batch_z})
                errD_real = self.D_real_loss.eval({self.inputs: batch_images})
                errG = self.G_loss.eval({self.z: batch_z})
                
                batch_d_loss = errD_fake + errD_real
                batch_g_loss = errG

                epoch_d_loss += batch_d_loss
                epoch_g_loss += batch_g_loss

                counter += 1
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"\
                      % (epoch, self.epochs, idx, batch_idxs,
                         time.time() - start_time, batch_d_loss, errG))

                if np.mod(counter, 500) == 2:
                    self.save(self.checkpoint_dir, counter)
            
            epoch_d_loss /= self.batch_size
            epoch_g_loss /= self.batch_size

            # Sample and evaluate inception score
            try:
                print("Sampling")
                sample_op = self.sampler(self.z)
                generated_images = self.sess.run(
                    sample_op,
                    feed_dict={
                        self.z: sample_z,
                    },
                )

                # Compute inception score
                print("Computing inception score")
                generated_images_list = [(image+1)*255/2 for image in generated_images]
                score = get_inception_score(generated_images_list, self.sess, splits=4)
                print(score)
                
                # Save images to png
                save_images(generated_images, image_manifold_size(generated_images.shape[0]), './{}/train_{:02d}_{:04d}.png'.format(self.sample_dir, epoch, idx))

            except Exception as e:
                print("Sampling error:", e)

            # Save results
            try:
                self.results['d_loss'].append(epoch_d_loss)
                self.results['g_loss'].append(epoch_g_loss)
                self.results['is'].append(score)

                with open(os.path.join(self.results_dir, 'output'), 'w') as of:
                    json.dump(self.results, of, cls=NumpyEncoder)
            except Exception as e:
                print("Result saving error:", e)
                
    
    def sampler(self, z, y=None):
        # TODO refactor with generator
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            s_h, s_w = self.image_shape
            s_h2, s_w2 = self.conv_out_size_same(s_h, 2), self.conv_out_size_same(s_w, 2)
            s_h4, s_w4 = self.conv_out_size_same(s_h2, 2), self.conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = self.conv_out_size_same(s_h4, 2), self.conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = self.conv_out_size_same(s_h8, 2), self.conv_out_size_same(s_w8, 2)

            # project `z` and reshape
            h0 = tf.reshape(
                linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
                [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0, train=False))

            h1 = deconv2d(h0, [self.sample_num, s_h8, s_w8, self.gf_dim*4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train=False))

            h2 = deconv2d(h1, [self.sample_num, s_h4, s_w4, self.gf_dim*2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=False))

            h3 = deconv2d(h2, [self.sample_num, s_h2, s_w2, self.gf_dim*1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train=False))

            h4 = deconv2d(h3, [self.sample_num, s_h, s_w, self.c_dim], name='g_h4')

            return tf.nn.tanh(h4)
        
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


