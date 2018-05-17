import tensorflow as tf
from model import DCGAN
from train import Trainer

def main(_):
    with tf.Session() as sess:
        dcgan = DCGAN(sess)
        trainer = Trainer(sess, dcgan)
        trainer.train()


if __name__ == '__main__':
    tf.app.run()
