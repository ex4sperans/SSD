import pytest
import tensorflow as tf

from datasets.voc_dataset import VOCDataset
from networks.vgg.vgg import VGG_16
from networks.ssd.ssd import SSD
from networks.ssd.ssd import TRAIN, INFERENCE
from networks.ssd.ssd import OutConvoLayer


def test_vgg_16():

    with tf.Graph().as_default():

        inputs = tf.placeholder(dtype=tf.float32, shape=(None, 300, 300, 3))

        vgg_16 = VGG_16((300, 300, 3), inputs)

        assert vgg_16.convo_output.get_shape().as_list() == [None, 9, 9, 512]

        with tf.Session() as sess:

            vgg_16.load_weights_from_npz(sess)


def test_ssd():

    class Config:

        input_shape = (300, 300, 3)
        weight_decay = 0.0005

        out_layers = [OutConvoLayer(name="out_convo4_3",
                                    parent="vgg_16.conv4_3",
                                    kernel_size=(3, 3),
                                    box_ratios=(1, 2, 1/2)),
                      OutConvoLayer(name="out_convo7",
                                    parent="conv7",
                                    kernel_size=(3, 3),
                                    box_ratios=(1, 2, 1/2))]

        classnames = VOCDataset.classnames

    with tf.Graph().as_default():

        ssd = SSD(Config(), mode=INFERENCE, resume=False)

        assert ssd.conv11_2.get_shape().as_list() == [None, 1, 1, 256]
