import functools
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from vgg.vgg import VGG_16
import misc
import preprocessing
import postprocessing
import boxes

class SSD:

    def __init__(self, model_params_path='ssd_params.json', resume=True):

        model_params = misc.load_json(model_params_path)

        self.input_shape = model_params['input_shape']
        self.class_names = model_params['class_names']
        self.feedforward_convo_architecture = model_params['feedforward_convo_architecture']
        self.out_convo_architecture = model_params['out_convo_architecture']
        self.scope = model_params['scope']
        self.first_layer = model_params['first_layer']
        self.model_path = model_params['model_path']

        self.vgg_16 = VGG_16(self.input_shape)
        self._create_graph()
        self._create_placeholders()
        self.model_vars = self._get_vars_by_scope(self.scope)
        self.saver = self._create_saver(self.model_vars + self.vgg_16.model_vars)

        if resume:
            self.load_model(verbose=True)
        else:
            self._init_vars(self.model_vars)
            self._init_vars(self.vgg_16.model_vars)
            self.vgg_16.load_convo_weights_from_npz(sess=self.sess)

        self.loss = self._create_loss()
        self._create_optimizer(self.loss)
        print('Number of parameters for {scope}: {n:.1f}M'.format(
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

            out_layers, self.out_shapes, self.box_ratios = [], [], []

            for layer_params in self.out_convo_architecture:
                (name, params), = layer_params.items()

                parent_layer = self._nested_getattr(params['parent'])

                #number of layer boxes times number of classes + 1 (background)
                #plus number of layer boxes times 4 (box correction)
                #layer boxes is the number of boxes per pixel of CNN feature map
                #usually is set to 2, 3 or 6
                layer_boxes = len(params['box_ratios'])
                self.box_ratios.append(params['box_ratios'])
                depth_per_box = self.n_classes + 1 + 4
                depth = layer_boxes*depth_per_box

                layer = slim.conv2d(
                                    parent_layer,
                                    depth,
                                    params['kernel_size'],
                                    params['stride'],
                                    scope=name)  

                setattr(self, name, layer)
                print('{name} with shape {shape}'.format(
                        name=name, shape=layer.get_shape()))
                height, width = misc.height_and_width(layer.get_shape().as_list())
                new_shape = (-1, height*width*layer_boxes, depth_per_box)
                out_layers.append(tf.reshape(layer, new_shape))
                self.out_shapes.append(tuple(layer.get_shape().as_list()))

            stacked_out_layers = tf.concat(1, out_layers)
            #slice stacked output along third dimension 
            #to obtain labels and offsets 
            self.total_boxes = stacked_out_layers.get_shape().as_list()[1]
            self.predicted_labels = tf.slice(stacked_out_layers, [0, 0, 0], [-1, -1, self.n_classes + 1])
            self.predicted_offsets = tf.slice(stacked_out_layers, [0, 0, self.n_classes + 1], [-1, -1, 4]) 
            print('\nPredicted labels shape: {shape}'.format(shape=self.predicted_labels.get_shape()))
            print('Predicted offsets shape: {shape}'.format(shape=self.predicted_offsets.get_shape()))

    def _create_graph(self):
        with tf.variable_scope(self.scope):
            with tf.variable_scope('feedforward_convo'):
                self._create_feedforward_convo()
            with tf.variable_scope('out_convo'):
                self._create_out_convo()

    def _create_placeholders(self):
        self.images = self.vgg_16.inputs
        self.labels = tf.placeholder(dtype=tf.int32, shape=(None, self.total_boxes))
        self.offsets = tf.placeholder(dtype=tf.float32, shape=(None, self.total_boxes, 4))
        self.positives = tf.placeholder(dtype=tf.float32, shape=(None, self.total_boxes))
        self.negatives = tf.placeholder(dtype=tf.float32, shape=(None, self.total_boxes))
        self.learning_rate = tf.placeholder(dtype=tf.float32)

    def _create_loss(self):
        positives_and_negatives = self.positives + self.negatives
        classification_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                self.predicted_labels, self.labels)
        classification_loss *= positives_and_negatives
        classification_loss = tf.reduce_sum(classification_loss, 1)
        classification_loss /= tf.reduce_sum(positives_and_negatives, 1) + 1e-6

        localization_loss = self._smooth_L1(self.predicted_offsets - self.offsets)
        localization_loss = tf.reduce_sum(localization_loss, 2)*self.positives
        localization_loss = tf.reduce_sum(localization_loss, 1)
        localization_loss /= tf.reduce_sum(self.positives, 1) + 1e-6

        self.classification_loss = tf.reduce_mean(classification_loss)
        self.localization_loss = tf.reduce_mean(localization_loss)

        #average over minibatch
        loss = tf.reduce_mean(classification_loss + localization_loss)
        return loss

    def _create_optimizer(self, loss):
        with tf.variable_scope('Optimizer_' + self.scope):
            optimizer = tf.train.AdamOptimizer(
                            learning_rate=self.learning_rate)
            grads_and_vars = optimizer.compute_gradients(loss)
            self.train_step = optimizer.apply_gradients(grads_and_vars)
        self.optimizer_vars = self._get_vars_by_scope('Optimizer_' + self.scope)
        self._init_vars(self.optimizer_vars)

    def _create_saver(self, vars_):
        return tf.train.Saver(vars_)

    def _nested_getattr(self, attr):
        #getattr built-in extended with capability of handling nested attributes
        return functools.reduce(getattr, attr.split('.'), self)

    def _smooth_L1(self, x):
        L2 = tf.square(x)/2
        L1 = tf.abs(x) - 0.5
        cond = tf.less(tf.abs(x), 1.0)
        distance = tf.select(cond, L2, L1)
        return distance

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

    @property
    def n_classes(self):
        return len(self.class_names)

    def _init_vars(self, vars_=None):
        self.sess.run(tf.variables_initializer(vars_))

    def save_model(self, path=None, sess=None, verbose=False):
        save_dir = path or self.model_path
        os.makedirs(save_dir, exist_ok=True)
        self.saver.save(sess or self.sess,
                        os.path.join(save_dir, 'model.ckpt'))
        if verbose:
            print('\nFollowing vars have been saved to {}:'.format(save_dir))
            for v in self.saver._var_list:
                print('{}.'.format(v.name))

    def load_model(self, path=None, sess=None, verbose=False):
        if path is None:
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            if ckpt is None:
                raise FileNotFoundError('Can`t load a model. '\
                'Checkpoint does not exist.')    
        restore_path = path or ckpt.model_checkpoint_path
        print('\nFollowing vars have been restored from {}:'.format(restore_path))
        self.saver.restore(sess or self.sess, restore_path)
        if verbose:
            for v in self.saver._var_list:
                print('{}'.format(v.name))

    def confidences_and_corrections(self, feed_dict):

        fetches = [tf.nn.softmax(self.predicted_labels), self.predicted_offsets]
        confidences, corrections = self.sess.run(fetches, feed_dict)
        return confidences, corrections

    def train(self, loader, overlap_threshold, neg_pos_ratio,
              batch_size, learning_rate, n_iter, test_freq, save_freq):

        default_boxes = boxes.get_default_boxes(self.out_shapes, self.box_ratios)

        for iteration in range(n_iter):
            train_batch = loader.new_train_batch(batch_size)
            images, offsets, labels = preprocessing.get_feed(
                            train_batch, self, default_boxes, overlap_threshold)
            feed_dict = {
                         self.offsets: offsets,
                         self.labels: labels,
                         self.images: images,
                         self.learning_rate: learning_rate}

            confidences, corrections = self.confidences_and_corrections(feed_dict)
            positives, negatives = preprocessing.positives_and_negatives(
                            confidences, labels, self, neg_pos_ratio)

            #add positives and negatives to feed dict
            feed_dict[self.positives] = positives
            feed_dict[self.negatives] = negatives

            fetches = [self.train_step, self.classification_loss, self.localization_loss]
            _, class_loss, loc_loss = self.sess.run(fetches, feed_dict)
            print('Iteration {}, Classificaion loss: {}, localization loss: {}'.format(
                                                        iteration, class_loss, loc_loss))

            if iteration % save_freq == 0:
                self.save_model(verbose=True)

            if iteration % test_freq == 0:
                test_batch = loader.new_test_batch(1)
                images, offsets, labels = preprocessing.get_feed(
                            test_batch, self, default_boxes, overlap_threshold)
                feed_dict = {
                             self.offsets: offsets,
                             self.labels: labels,
                             self.images: images}

                confidences, corrections = self.confidences_and_corrections(feed_dict)
        
                postprocessing.draw_top_boxes(
                                              batch=test_batch,
                                              confidences=confidences,
                                              corrections=corrections,
                                              default_boxes=default_boxes,
                                              threshold=overlap_threshold,
                                              save_path='predictions',
                                              iteration=iteration,
                                              model=self)