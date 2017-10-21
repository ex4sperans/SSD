import functools
import os
import json

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from vgg.vgg import VGG_16
from multithreaded_data_provider import MultithreadedTensorProvider

import misc
import preprocessing
import postprocessing
import boxes

class SSD:

    def __init__(self, model_params_path='ssd_params.json', resume=True, mode='train'):

        model_params = misc.load_json(model_params_path)

        self.input_shape = model_params['input_shape']
        self.class_names = model_params['class_names']
        self.feedforward_convo_architecture = model_params['feedforward_convo_architecture']
        self.out_convo_architecture = model_params['out_convo_architecture']
        self.scope = model_params['scope']
        self.first_layer = model_params['first_layer']
        self.model_path = model_params['model_path']
        self.neg_pos_ratio = model_params['neg_pos_ratio']

        self.mode = mode
        self._create_placeholders()
        self._create_graph()
        self._create_step()

        if mode == 'train':
            self.loss = self._create_loss()
            self.optimizer_vars = self._create_optimizer(self.loss)
            
        self.model_vars = self._get_vars_by_scope(self.scope)

        if mode == 'train':
            self.saver_vars = self.model_vars + self.vgg_16.model_vars + self.optimizer_vars
        elif mode == 'inference':
            self.saver_vars = self.model_vars + self.vgg_16.model_vars

        self.saver = self._create_saver(self.saver_vars)

        if resume:
            self.load_model(verbose=True)
        else:
            self._init_vars(self.model_vars)
            self._init_vars(self.vgg_16.model_vars)
            self._init_vars(self.optimizer_vars)
            self.vgg_16.load_convo_weights_from_npz(sess=self.sess)

        print('Number of parameters for {scope}: {n:.1f}M'.format(
            scope=self.scope, n=self._number_of_parameters(self.model_vars)/1e6))

    def _create_feedforward_convo(self):

        layer = self._nested_getattr(self.first_layer)

        with slim.arg_scope([slim.conv2d],
              activation_fn=None,
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
                layer = slim.batch_norm(
                                        layer,
                                        scale=True,
                                        updates_collections=None,
                                        is_training=self.is_training,
                                        scope='batch_norm_{}'.format(name))
                layer = tf.nn.relu(layer)
                
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

                # number of layer boxes times number of classes + 1 (background)
                # plus number of layer boxes times 4 (box correction)
                # layer boxes is the number of boxes per cell of CNN feature map
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
            # slice stacked output along third dimension 
            # to obtain labels and offsets 
            self.total_boxes = stacked_out_layers.get_shape().as_list()[1]
            self.predicted_labels = tf.slice(stacked_out_layers, [0, 0, 0], [-1, -1, self.n_classes + 1])
            self.predicted_offsets = tf.slice(stacked_out_layers, [0, 0, self.n_classes + 1], [-1, -1, 4]) 
            self.confidences = tf.nn.softmax(self.predicted_labels)
            print('\nPredicted labels shape: {shape}'.format(shape=self.predicted_labels.get_shape()))
            print('Predicted offsets shape: {shape}'.format(shape=self.predicted_offsets.get_shape()))

    def _create_graph(self):
        with tf.variable_scope(self.scope):
            with tf.variable_scope('feedforward_convo'):
                self._create_feedforward_convo()
            with tf.variable_scope('out_convo'):
                self._create_out_convo()

    def _create_placeholders(self):

        self.learning_rate = tf.placeholder(dtype=tf.float32)
        self.test_images = tf.placeholder(dtype=tf.float32, shape=[None] + self.input_shape)
        self.is_training = tf.placeholder(dtype=tf.bool, shape=())              
        if self.mode == 'train':
            self.tensor_provider = MultithreadedTensorProvider(
                                                capacity=20,
                                                sess=self.sess,
                                                dtypes=[tf.float32, tf.int32, tf.float32],
                                                number_of_threads=2)

            self.train_images, self.labels, self.offsets = self.tensor_provider.get_input()
            self.images = tf.cond(
                                  self.is_training,
                                  lambda: self.train_images,
                                  lambda: self.test_images)
        elif self.mode == 'inference':
            self.images = self.test_images
        else:
            raise ValueError('Incorrect mode: {}'.format(self.mode))
        self.vgg_16 = VGG_16(self.input_shape, self.images)

    def _create_loss(self):
    
        self.positives, self.negatives = self._positives_and_negatives(
                    self.confidences, self.labels, self.neg_pos_ratio)
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
        self.l2_loss = tf.add_n(slim.losses.get_regularization_losses())
        # average over minibatch
        loss = self.classification_loss + self.localization_loss + self.l2_loss
        return loss

    def _positives_and_negatives(self, confidences, labels, neg_pos_ratio):

        background_class = self.n_classes
        one_hot_labels = tf.one_hot(labels, self.n_classes + 1, axis=2)
        positives = tf.cast(tf.not_equal(labels, background_class), tf.float32)
        n_positives = tf.reduce_sum(positives, 1, keep_dims=True)
        n_negatives = tf.cast(n_positives*neg_pos_ratio, tf.float32)
        true_labels_mask = tf.cast(tf.logical_not(tf.cast(one_hot_labels, tf.bool)), tf.float32)
        top_wrong_confidences = tf.reduce_max(confidences*true_labels_mask, axis=2)
        non_positive_mask = tf.cast(tf.logical_not(tf.cast(positives, tf.bool)), tf.float32)

        def get_threshold(inputs):
            conf = tf.slice(inputs, [0], [self.total_boxes])
            k = tf.slice(inputs, [self.total_boxes], [1])
            k = tf.squeeze(tf.cast(k, tf.int32), 0)
            top, _ = tf.nn.top_k(conf, k)

            return tf.cond(
                           tf.greater(k, 0),
                           lambda: tf.slice(top, [k-1], [1]),
                           lambda: tf.constant([1.0]))

        map_inputs = tf.concat(1, (top_wrong_confidences*non_positive_mask, n_negatives))
        thresholds = tf.map_fn(get_threshold, map_inputs)
        self.thresholds = thresholds
        negatives = tf.cast(
            tf.greater(top_wrong_confidences*non_positive_mask, thresholds), tf.float32)

        return positives, negatives

    def _create_optimizer(self, loss):
        self.learning_rate = tf.placeholder(dtype=tf.float32)

        with tf.variable_scope('Optimizer_' + self.scope):
            optimizer = tf.train.MomentumOptimizer(
                            learning_rate=self.learning_rate,
                            momentum=0.9)
            grads_and_vars = optimizer.compute_gradients(loss)
            self.train_step = optimizer.apply_gradients(grads_and_vars,
                                                        global_step=self.step)
        optimizer_vars = self._get_vars_by_scope('Optimizer_' + self.scope)
        return optimizer_vars

    def _create_saver(self, vars_):
        return tf.train.Saver(vars_)        

    def _create_step(self):
        with tf.variable_scope(self.scope):
            self.step = tf.get_variable(name='step', 
                initializer=tf.ones(shape=(), dtype=tf.int32),
                trainable=False)
            self.sess.run(tf.variables_initializer([self.step]))

    def get_step(self):
        return self.sess.run(self.step)

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
            config.inter_op_parallelism_threads = 8
            config.intra_op_parallelism_threads = 8
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
        fetches = [self.confidences, self.predicted_offsets]
        confidences, corrections = self.sess.run(fetches, feed_dict)
        return confidences, corrections

    def train(self, loader, overlap_threshold, nms_threshold, neg_pos_ratio,
              batch_size, learning_rate_schedule, n_iter, test_freq, save_freq):

        default_boxes = boxes.get_default_boxes(self.out_shapes, self.box_ratios)

        def data_provider():

            train_batch = loader.new_train_batch(batch_size, augment=True)
            images, offsets, labels = preprocessing.get_feed(
                        train_batch, self, default_boxes, overlap_threshold)

            return (np.array(images, dtype=np.float32),
                    np.array(labels, dtype=np.int32),
                    np.array(offsets, dtype=np.float32))            

        self.tensor_provider.set_data_provider(data_provider)

        for iteration in range(self.get_step(), n_iter):

            learning_rate = learning_rate_schedule(iteration)
            
            self.train_iteration(
                                 iteration=iteration,
                                 learning_rate=learning_rate,
                                 default_boxes=default_boxes)

            if iteration % save_freq == 0:
                self.save_model()

            if iteration % test_freq == 0:
                test_batch = loader.new_train_batch(1, augment=False)
                self.test_iteration(
                                    test_batch=test_batch,
                                    default_boxes=default_boxes,
                                    overlap_threshold=overlap_threshold,
                                    nms_threshold=nms_threshold,
                                    iteration=iteration,
                                    save_path='predictions/train')                                    

                test_batch = loader.new_test_batch(1)
                self.test_iteration(
                                    test_batch=test_batch,
                                    default_boxes=default_boxes,
                                    overlap_threshold=overlap_threshold,
                                    nms_threshold=nms_threshold,
                                    iteration=iteration,
                                    save_path='predictions/test')

    def train_iteration(self, iteration, learning_rate, default_boxes):

        feed_dict = {
                     # fake input to fill a test placeholder
                     self.test_images: np.zeros([1] + self.input_shape),
                     self.learning_rate: learning_rate,
                     self.is_training: True}

        fetches = [self.train_step, self.classification_loss, self.localization_loss]
        _, class_loss, loc_loss = self.sess.run(fetches, feed_dict)

        # fetches = [self.train_step, self.classification_loss, self.localization_loss, self.images, self.positives]
        # _, class_loss, loc_loss, images, positives = self.sess.run(fetches, feed_dict)

        # default_boxes = misc.flatten_list(default_boxes)
        # for i, (image, image_positives) in enumerate(zip(images, positives)):

        #     matched_boxes = [default_box for default_box, pos in zip(default_boxes, image_positives) if pos]
        #     height, width = misc.height_and_width(self.input_shape)
        #     matched_boxes = [boxes.recover_box(box, height, width) for box in matched_boxes]

        #     boxes.plot_with_bboxes(image, 'batch_plots', str(i) + '.jpg', matched_boxes, [])


        print('Iteration {}, Classificaion loss: {}, localization loss: {}'.format(
                                                        iteration, class_loss, loc_loss))

    def test_iteration(self, test_batch, default_boxes, overlap_threshold,
                       nms_threshold, iteration, save_path):

        images, offsets, labels = preprocessing.get_feed(
                            test_batch, self, default_boxes, overlap_threshold)
        feed_dict = {
                     self.test_images: images,
                     self.is_training: False}

        confidences, corrections = self.confidences_and_corrections(feed_dict)

        postprocessing.draw_top_boxes(
                                      batch=test_batch,
                                      confidences=confidences,
                                      corrections=corrections,
                                      default_boxes=default_boxes,
                                      threshold=nms_threshold,
                                      save_path=save_path,
                                      iteration=iteration,
                                      model=self)


    def predict_single_image(self, image_path, predicted_image_path, 
                             nms_threshold):

        default_boxes = boxes.get_default_boxes(self.out_shapes, self.box_ratios)
        image = misc.load_image(image_path)
        resized_image = misc.resize(image, self.input_shape)
        original_height, original_width = misc.height_and_width(image.shape)
        height, width = misc.height_and_width(self.input_shape)

        feed_dict = {
                     self.test_images: [resized_image/255],
                     self.is_training: False}

        confidences, corrections = self.confidences_and_corrections(feed_dict)
        confidences, corrections = confidences[0], corrections[0]

        bboxes, labels, confidences = postprocessing.non_maximum_supression(
                                            confidences=confidences,
                                            default_boxes=default_boxes,
                                            corrections=corrections,
                                            class_names=self.class_names,
                                            threshold=nms_threshold,
                                            height=original_height,
                                            width=original_width)

        boxes.plot_predicted_bboxes(
                                    image=image,
                                    save_path='./',
                                    file_name=predicted_image_path,
                                    bboxes=bboxes,
                                    labels=labels,
                                    confidences=confidences,
                                    class_names=self.class_names)