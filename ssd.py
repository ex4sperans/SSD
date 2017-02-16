import json
import functools

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from vgg.vgg import VGG_16

class SSD:

    def __init__(self, model_params_path='ssd_params.json'):

        model_params = self._load_params_json(model_params_path)

        self.input_shape = model_params['input_shape']
        self.class_names = model_params['class_names']
        self.feedforward_convo_architecture = model_params['feedforward_convo_architecture']
        self.out_convo_architecture = model_params['out_convo_architecture']
        self.scope = model_params['scope']
        self.first_layer = model_params['first_layer']

        self.vgg_16 = VGG_16(self.input_shape, self.class_names)
        self._create_graph()

        self.model_vars = self._get_vars_by_scope(self.scope)
        print('\nNumber of parameters for {scope}: {n:.1f}M'.format(
            scope=self.scope, n=self._number_of_parameters(self.model_vars)/1e6))

    def _create_feedforward_convo(self):

        layer = self._nested_getattr(self.first_layer)

        with slim.arg_scope([slim.conv2d],
              activation_fn=tf.nn.relu,
              weights_regularizer=slim.l2_regularizer(0.0005)):

            print('\nCreating layers for {scope}:'.format(scope=self.scope))

            for layer_params in self.feedforward_convo_architecture:
                (name, params), = layer_params.items()
                if name.startswith('conv'):
                    layer = slim.conv2d(
                                        layer,
                                        params['depth'],
                                        params['kernel_size'],
                                        params['stride'],
                                        scope=name) 
                elif name.startswith('avg_pool'):
                    layer = slim.avg_pool2d(
                                            layer,
                                            params['kernel_size'],
                                            padding=params['padding'],
                                            scope=name) 
                setattr(self, name, layer)
                print('{name} with shape {shape}'.format(
                        name=name, shape=layer.get_shape()))

    def _create_out_convo(self):

        with slim.arg_scope([slim.conv2d],
              activation_fn=None,
              weights_regularizer=slim.l2_regularizer(0.0005)):

            print('\nCreating layers for {scope}:'.format(scope=self.scope))

            for layer_params in self.out_convo_architecture:
                (name, params), = layer_params.items()

                parent_layer = self._nested_getattr(params['parent'])

                #number of layer boxes times number of classes + 1 (background)
                #plus number of layer boxes times 4 (box correction)
                #layer boxes is the number of boxes per pixel of CNN feature map
                #usually is set to 2, 3 or 6
                depth = params['layer_boxes']*(self.n_classes + 1 + 4)
                
                layer = slim.conv2d(
                                    parent_layer,
                                    depth,
                                    params['kernel_size'],
                                    params['stride'],
                                    scope=name)  
                setattr(self, name, layer)
                print('{name} with shape {shape}'.format(
                        name=name, shape=layer.get_shape()))

    def _create_graph(self):
        with tf.variable_scope(self.scope):
            with tf.variable_scope('feedforward_convo'):
                self._create_feedforward_convo()
            with tf.variable_scope('out_convo'):
                self._create_out_convo()


    def _nested_getattr(self, attr):
        return functools.reduce(getattr, attr.split('.'), self)

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

    @property
    def n_classes(self):
        return len(self.class_names)

