import tensorflow as tf
import numpy as np
import time
import os
import json
import config
from utils import save_images, NumpyEncoder, image_manifold_size
from inception_score import get_inception_score

class Trainer(object):
    def __init__(self, sess, model):
        self.model = model
        self.sess = sess
        self.results = {
            "d_loss": [],
            "g_loss": [],
            "is": []
        }
        self.results_dir = config.DATA['results_dir']
            
    def train(self):
            d_optim = tf.train.AdamOptimizer(self.model.learning_rate, beta1=self.model.beta1).minimize(self.model.D_loss, var_list=self.model.d_vars)
            g_optim = tf.train.AdamOptimizer(self.model.learning_rate, beta1=self.model.beta1).minimize(self.model.G_loss, var_list=self.model.g_vars)
            global_step = tf.Variable(0, name="global_step", trainable=False)

            try:
                tf.global_variables_initializer().run()
            except:
                tf.initialize_all_variables().run()

            self.model.sample_z = np.random.uniform(-1, 1, size=(self.model.sample_num, self.model.z_dim))

            increase_global_step = global_step.assign(global_step + 1)
            self.model.sess.run(global_step)

            counter = 1
            start_time = time.time()
            could_load, checkpoint_counter = self.load(self.model.checkpoint_dir)
            if could_load:
                counter = checkpoint_counter
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

            # Epoch loop
            for epoch in range(self.model.epochs):
                epoch_d_loss = 0
                epoch_g_loss = 0
                for idx in range(0, self.model.batch_idxs):
                    batch = self.model.data[idx * self.model.batch_size:(idx + 1) * self.model.batch_size]

                    if self.model.dataset_name == 'reptiles':
                        batch = [get_image(batch_file,
                                input_height=self.model.image_shape[0],
                                input_width=self.model.image_shape[1],
                                resize_height=self.model.image_shape[0],
                                resize_width=self.model.image_shape[1]) for batch_file in batch]

                    batch_images = np.array(batch).astype(np.float32)
                    batch_z = np.random.uniform(-1, 1, [self.model.batch_size, self.model.z_dim]) \
                        .astype(np.float32)

                    # Update networks and spectral norm
                    self.model.sess.run([d_optim], feed_dict={self.model.inputs: batch_images, self.model.z: batch_z})
                    self.model.sess.run([g_optim], feed_dict={self.model.z: batch_z}) # We used to run this line twice
                    if self.model.use_spectral_norm:
                        for update_op in self.model.sn_update_ops:
                            self.model.sess.run(update_op)

                    # Compute loss
                    errD_fake = self.model.D_fake_loss.eval({self.model.z: batch_z})
                    errD_real = self.model.D_real_loss.eval({self.model.inputs: batch_images})
                    errG = self.model.G_loss.eval({self.model.z: batch_z})
            
                    # Save loss
                    batch_d_loss = errD_fake + errD_real
                    batch_g_loss = errG
                    epoch_d_loss += batch_d_loss
                    epoch_g_loss += batch_g_loss

                    # Print progress
                    counter += 1

                    current_global_step = self.model.sess.run(increase_global_step)
                    print("Global step: %2d\tEpoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"\
                        % (current_global_step, epoch, self.model.epochs, idx, self.model.batch_idxs,
                            time.time() - start_time, batch_d_loss, errG))

                    # Save model
                    if np.mod(counter, 500) == 2:
                        self.save(self.model.checkpoint_dir, counter)
                
                # Compute epoch loss
                epoch_d_loss /= self.model.batch_size
                epoch_g_loss /= self.model.batch_size

                # Sample and evaluate inception score
                print("Generating images and computing inception score")
                score = self.compute_inception_score(epoch, idx)
                print(score)

                # Save results
                self.save_results(epoch_d_loss, epoch_g_loss, score)

    def save_results(self, epoch_d_loss, epoch_g_loss, score):
        # Appends to results dict and rewrites it
        try:
            self.results['d_loss'].append(epoch_d_loss)
            self.results['g_loss'].append(epoch_g_loss)
            self.results['is'].append(score)
            with open(os.path.join(self.results_dir, 'output'), 'w') as of:
                json.dump(self.results, of, cls=NumpyEncoder)

        except Exception as e:
            print("Result saving error:", e)

    def compute_inception_score(self, epoch, idx):
        # Generates images and their inception score
        try:
            # Generate images and save them
            sample_op = self.model.generator(self.model.z, train=False)
            generated_images = self.sess.run(
                sample_op,
                feed_dict={
                    self.model.z: self.model.sample_z,
                },
            ) 
            save_images(generated_images, image_manifold_size(generated_images.shape[0]), './{}/train_{:02d}_{:04d}.png'.format(self.model.sample_dir, epoch, idx))

            # Compute inception score
            generated_images_list = [(image+1)*255/2 for image in generated_images]
            score = get_inception_score(generated_images_list, self.sess, splits=5)
            
            return score

        except Exception as e:
            print("Sampling error:", e)
            return np.nan
        
    def save(self, checkpoint_dir, step):
        # TODO make sure we dont overwrite existing models.
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.model.saver.save(self.model.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.model.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0