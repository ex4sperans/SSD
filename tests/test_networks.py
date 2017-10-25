import tensorflow as tf
import pytest

from networks.vgg.vgg import VGG_16


def test_vgg_16():

    with tf.Graph().as_default():

        inputs = tf.placeholder(dtype=tf.float32, shape=(None, 300, 300, 3))

        vgg_16 = VGG_16((300, 300, 3), inputs)

        assert vgg_16.convo_output.get_shape().as_list() == [None, 9, 9, 512]

        with tf.Session() as sess:

            vgg_16.load_weights_from_npz(sess)