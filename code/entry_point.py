import tensorflow as tf
from model import DCGAN


def main(_):
    with tf.Session() as sess:
        dcgan = DCGAN(sess)
        dcgan.train()

        # Below is codes for visualization
        # visualize(sess, dcgan, FLAGS, OPTION)


if __name__ == '__main__':
    tf.app.run()
