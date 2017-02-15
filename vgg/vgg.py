import os
from operator import attrgetter

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim


class VGG_16:

    def __init__(self, input_shape, class_names, scope, convo_architecture):

        self.input_shape = input_shape
        self.class_names = class_names
        self.convo_architecture = convo_architecture
        self.scope = scope

        self._create_placeholders()
        self._create_graph()

    def _create_placeholders(self):

        self.inputs = tf.placeholder(tf.float32, shape=[None] + self.input_shape)
        self.labels = tf.placeholder(tf.float32, shape=[None, len(self.class_names)])

    def _convo_layers(self, inputs):
        
        with slim.arg_scope([slim.conv2d],
                  activation_fn=tf.nn.relu,
                  weights_regularizer=slim.l2_regularizer(0.0005)):

            print('\nCreating layers for {scope}:'.format(scope=self.scope))

            layer = inputs
            for layer_params in self.convo_architecture:
                (name, params), = layer_params.items()
                if name.startswith('conv'):
                    layer = slim.conv2d(
                                        layer,
                                        params['depth'],
                                        params['kernel_size'],
                                        scope=name) 
                elif name.startswith('pool'):
                    layer = slim.max_pool2d(
                                            layer,
                                            params['kernel_size'],
                                            scope=name) 
                setattr(self, name, layer)
                print('{name} with {shape}'.format(name=name, shape=layer.get_shape()))

        return layer

    def _create_graph(self):
        with tf.variable_scope(self.scope):
            with tf.variable_scope('convo_layers'):
                self.convo_output = self._convo_layers(self.inputs)

    def load_convo_weights(self, model_path='vgg16_weights.npz', sess=None):

        sess = sess or self.sess
        saved_weights = np.load(model_path)
        #sort saved weights and tf variables to make them match each other
        keys = sorted(saved_weights.keys(), key=str.lower)
        keys = [k for k in keys if k.startswith('conv')]
        convo_vars = self._get_vars_by_scope(self.scope + '/convo_layers')
        convo_vars = sorted(convo_vars, key=attrgetter('name'))

        print('\nLoading weights for {scope}:'.format(scope=self.scope))
        for key, var in zip(keys, convo_vars):
            saved_var = saved_weights[key]
            print('{name} with shape {shape}'.format(name=key, shape=saved_var.shape))
            self.sess.run(var.assign(saved_var))

    @property
    def sess(self):
        if not hasattr(self, '_sess'):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self._sess = tf.Session(config=config)
        return self._sess

    def _get_vars_by_scope(self, scope, only_trainable=False):
        if only_trainable:
            vars_ = tf.trainable_variables()
        else:
            vars_ = tf.global_variables()

        return list(v for v in vars_ if v.name.startswith(scope))