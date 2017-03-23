import os
from operator import attrgetter
import json

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

class VGG_16:

    def __init__(self, input_shape, inputs, model_params_path='vgg/vgg_params.json'):

        model_params = self._load_params_json(model_params_path)

        self.input_shape = input_shape
        self.convo_architecture = model_params['convo_architecture']
        self.scope = model_params['scope']
        self.blue_mean, self.green_mean, self.red_mean = model_params['mean']

        self.inputs = tf.reshape(inputs, [-1] + self.input_shape)
        self._create_graph()
        self.model_vars = self._get_vars_by_scope(self.scope)

        print('\nNumber of parameters for {scope}: {n:.1f}M'.format(
            scope=self.scope, n=self._number_of_parameters(self.model_vars)/1e6))

    def _create_convo_layers(self, inputs):

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
                                            padding='SAME',
                                            scope=name) 
                setattr(self, name, layer)
                print('{name} with shape {shape}'.format(name=name, shape=layer.get_shape()))

    def _preprocess(self, inputs):
        #inputs assumed to be a RGB image
        rgb_scaled = inputs * 255.0
        subtracted = rgb_scaled - tf.constant([
                                               self.red_mean,
                                               self.green_mean,
                                               self.blue_mean],
                                               dtype=tf.float32,
                                               shape=[1, 1, 1, 3])
        return subtracted

    def _create_graph(self):
        with tf.variable_scope(self.scope):
            with tf.variable_scope('convo_layers'):
                inputs = self._preprocess(self.inputs)
                self.convo_output = self._create_convo_layers(inputs)

    def load_convo_weights_from_npz(self, model_path='vgg/vgg16_weights.npz', sess=None):

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

    def _number_of_parameters(self, vars_):
        return sum(np.prod(v.get_shape().as_list()) for v in vars_)

    def _load_params_json(self, json_path):
        with open(json_path, 'r') as f:
            params = json.load(f)
        return params
