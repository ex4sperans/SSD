import os
from operator import attrgetter

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim


# ImageNet mean values
BLUE = 103.939
GREEN = 116.779
RED = 123.68

SAVED_WEIGHTS = os.path.join(os.path.dirname(__file__), "vgg16_weights.npz")


class VGG_16:

    def __init__(self, input_shape, inputs, trainable=True):

        self.input_shape = input_shape
        self.scope = "VGG_16"
        self.inputs = tf.reshape(inputs, (-1,) + self.input_shape)
        self.trainable = trainable

        self._create_graph()
        self.model_vars = self._get_vars_by_scope(self.scope)

    def _create_convo_layers(self, inputs):

        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            kernel_size=(3, 3),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            trainable=self.trainable):

            self.conv1_1 = slim.conv2d(inputs, 64, scope='conv1_1')
            self.conv1_2 = slim.conv2d(self.conv1_1, 64, scope='conv1_2')
            self.pool_1 = slim.max_pool2d(self.conv1_2, (2, 2), scope='pool_1')

            self.conv2_1 = slim.conv2d(self.pool_1, 128, scope='conv2_1')
            self.conv2_2 = slim.conv2d(self.conv2_1, 128, scope='conv2_2')
            self.pool_2 = slim.max_pool2d(self.conv2_2, (2, 2), scope='pool_2')

            self.conv3_1 = slim.conv2d(self.pool_2, 256, scope='conv3_1')
            self.conv3_2 = slim.conv2d(self.conv3_1, 256, scope='conv3_2')
            self.conv3_3 = slim.conv2d(self.conv3_2, 256, scope='conv3_3')
            self.pool_3 = slim.max_pool2d(self.conv3_3, (2, 2), scope='pool_3')

            self.conv4_1 = slim.conv2d(self.pool_3, 512, scope='conv4_1')
            self.conv4_2 = slim.conv2d(self.conv4_1, 512, scope='conv4_2')
            self.conv4_3 = slim.conv2d(self.conv4_2, 512, scope='conv4_3')
            self.pool_4 = slim.max_pool2d(self.conv4_3, (2, 2), scope='pool_4')

            self.conv5_1 = slim.conv2d(self.pool_4, 512, scope='conv5_1')
            self.conv5_2 = slim.conv2d(self.conv5_1, 512, scope='conv5_2')
            self.conv5_3 = slim.conv2d(self.conv5_2, 512, scope='conv5_3')
            self.pool_5 = slim.max_pool2d(self.conv5_3, (2, 2), scope='pool_5')

            return self.pool_5

    def _preprocess(self, inputs):
        # inputs assumed to be normalized RGB images
        rgb_scaled = inputs * 255.0
        subtracted = rgb_scaled - tf.constant([RED,
                                               GREEN,
                                               BLUE],
                                              dtype=tf.float32,
                                              shape=(1, 1, 1, 3))
        return subtracted

    def _create_graph(self):
        with tf.variable_scope(self.scope):
            with tf.variable_scope('convo_layers'):
                inputs = self._preprocess(self.inputs)
                self.convo_output = self._create_convo_layers(inputs)

    def load_weights_from_npz(self, sess):

        saved_weights = np.load(SAVED_WEIGHTS)
        # sort saved weights and tf variables to make them match each other
        keys = sorted(saved_weights.keys(), key=str.lower)
        keys = [k for k in keys if k.startswith('conv')]
        convo_vars = self._get_vars_by_scope(self.scope + '/convo_layers')
        convo_vars = sorted(convo_vars, key=attrgetter('name'))

        for key, var in zip(keys, convo_vars):
            sess.run(var.assign(saved_weights[key]))

    def _get_vars_by_scope(self, scope, only_trainable=False):
        if only_trainable:
            var_list = tf.trainable_variables()
        else:
            var_list = tf.global_variables()

        return list(v for v in var_list if scope in v.name)
