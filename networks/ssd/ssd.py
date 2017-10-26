import collections
import functools
import os
import json

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from networks.vgg.vgg import VGG_16
from ops.multithreaded_data_provider import TensorProvider
from ops.misc import height_and_width


OutConvoLayer = collections.namedtuple("OutConvoLayer",
                                       ["name",
                                        "parent",
                                        "kernel_size",
                                        "box_ratios"])


TRAIN = "train"
INFERENCE = "inference"
MODES = (TRAIN, INFERENCE)


class SSD:

    def __init__(self, config, mode, resume=True):

        self.config = config
        self.scope = "SSD"

        if not mode in MODES:
            raise ValueError("`mode` should be one of {}".format(MODES))

        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope(self.scope):

                self.mode = mode
                self._create_placeholders()
                self._create_graph()
                self._create_step()

                self.model_vars = self._get_vars_by_scope(self.scope)
                self.all_vars = self.model_vars + self.vgg_16.model_vars

                if mode is TRAIN:
                    self.loss = self._create_loss()
                    self._create_optimizer(self.loss)
                    self.optimizer_vars = self._get_vars_by_scope("optimizer")
                    self.all_vars += self.optimizer_vars

                self.saver = self._create_saver(self.all_vars)

                if resume:
                    self.load_model(verbose=True)
                else:
                    self._init_vars(self.all_vars)
                    self.vgg_16.load_weights_from_npz(sess=self.sess)

    def _create_placeholders(self):

        self.is_training = tf.placeholder(dtype=tf.bool)

        self.test_images = tf.placeholder(
                                    dtype=tf.float32,
                                    shape=(None,) + self.config.input_shape)
        if self.mode is TRAIN:
            self.tensor_provider = TensorProvider(capacity=20,
                                                  sess=self.sess,
                                                  # images, labels, offsets
                                                  dtypes=(tf.float32,
                                                          tf.int32,
                                                          tf.float32),
                                                  number_of_threads=2)

            (self.train_images,
             self.labels,
             self.offsets) = self.tensor_provider.get_input()

            # make images to be output of tensor provider on train
            # and placeholder on test
            self.images = tf.cond(self.is_training,
                                  lambda: self.train_images,
                                  lambda: self.test_images)

        elif self.mode is INFERENCE:
            # just use placeholder as inputs on inference
            self.images = self.test_images

        self.vgg_16 = VGG_16(self.config.input_shape, self.images)

    def _create_graph(self):

        self.regularizer = slim.l2_regularizer(self.config.weight_decay)
    
        with tf.variable_scope('feedforward_convo'):
            self._create_feedforward_convo()
        with tf.variable_scope('out_convo'):
            self._create_out_convo()

    def _batch_norm(self, net, scope):

        return slim.batch_norm(net,
                               scale=True,
                               updates_collections=None,
                               is_training=self.is_training,
                               scope=scope)

    def _create_feedforward_convo(self):

        with slim.arg_scope([slim.conv2d],
                            activation_fn=None,
                            weights_regularizer=self.regularizer):

            net = self._batch_norm(self.vgg_16.conv5_3, "batch_norm_5_3")
            net = slim.conv2d(net, 1024, (3, 3), scope="conv6")
            net = self._batch_norm(net, "batch_norm6")
            self.conv6 = tf.nn.relu(net)

            net = slim.conv2d(self.conv6, 1024, (1, 1), scope="conv7")
            net = self._batch_norm(net, "batch_norm7")
            self.conv7 = tf.nn.relu(net)

            net = slim.conv2d(self.conv6, 256, (1, 1), scope="conv8_1")
            net = self._batch_norm(net, "batch_norm8_1")
            self.conv8_1 = tf.nn.relu(net)

            net = slim.conv2d(self.conv8_1, 512, (3, 3), 2, scope="conv8_2")
            net = self._batch_norm(net, "batch_norm8_2")
            self.conv8_2 = tf.nn.relu(net)

            net = slim.conv2d(self.conv8_2, 128, (1, 1), scope="conv9_1")
            net = self._batch_norm(net, "batch_norm9_1")
            self.conv9_1 = tf.nn.relu(net)

            net = slim.conv2d(self.conv9_1, 256, (3, 3), 2, scope="conv9_2")
            net = self._batch_norm(net, "batch_norm9_2")
            self.conv9_2 = tf.nn.relu(net)

            net = slim.conv2d(self.conv9_2, 128, (1, 1), scope="conv10_1")
            net = self._batch_norm(net, "batch_norm10_1")
            self.conv10_1 = tf.nn.relu(net)

            net = slim.conv2d(self.conv10_1, 256, (3, 3),
                              scope="conv10_2", padding="VALID")
            net = self._batch_norm(net, "batch_norm10_2")
            self.conv10_2 = tf.nn.relu(net)

            net = slim.conv2d(self.conv10_2, 128, (1, 1), scope="conv11_1")
            net = self._batch_norm(net, "batch_norm11_1")
            self.conv11_1 = tf.nn.relu(net)

            net = slim.conv2d(self.conv11_1, 256, (3, 3),
                              scope="conv11_2", padding="VALID")
            net = self._batch_norm(net, "batch_norm11_2")
            self.conv11_2 = tf.nn.relu(net)

    def _create_out_convo(self):

        with slim.arg_scope([slim.conv2d],
                            activation_fn=None,
                            weights_regularizer=self.regularizer):

            out_layers, self.out_shapes, self.box_ratios = [], [], []

            for layer in self.config.out_layers:

                parent_layer = self._nested_getattr(layer.parent)

                layer_boxes = len(layer.box_ratios)
                self.box_ratios.append(layer.box_ratios)
                # number of layer boxes times number of classes + 1 (background)
                # plus number of layer boxes times 4 (box offsets)
                # layer boxes is the number of boxes per cell of CNN feature map
                depth_per_box = self.n_classes + 1 + 4
                depth = layer_boxes * depth_per_box

                out = slim.conv2d(parent_layer,
                                  depth,
                                  layer.kernel_size,
                                  scope=layer.name)

                shape = tuple(out.get_shape().as_list())
                self.out_shapes.append(shape)

                height, width = height_and_width(shape)
                new_shape = (-1, height * width * layer_boxes, depth_per_box)
                out_layers.append(tf.reshape(out, new_shape))

                print("{name} with shape {shape}"
                      .format(name=layer.name, shape=shape))

            stacked_out_layers = tf.concat(out_layers, 1)
            self.total_boxes = stacked_out_layers.get_shape().as_list()[1]
            # slice stacked output along third dimension
            # to obtain labels and offsets
            self.predicted_labels = tf.slice(stacked_out_layers,
                                             [0, 0, 0],
                                             [-1, -1, self.n_classes + 1])
            self.predicted_offsets = tf.slice(stacked_out_layers,
                                              [0, 0, self.n_classes + 1],
                                              [-1, -1, 4])
            self.confidences = tf.nn.softmax(self.predicted_labels)

            print("\nPredicted labels shape: {shape}"
                  .format(shape=self.predicted_labels.get_shape()))
            print("Predicted offsets shape: {shape}"
                  .format(shape=self.predicted_offsets.get_shape()))


    def _create_loss(self):

        (self.positives,
         self.negatives) = self._positives_and_negatives(
                                                self.confidences,
                                                self.labels,
                                                self.config.neg_pos_ratio)
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

            return tf.cond(tf.greater(k, 0),
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

        with tf.variable_scope("optimizer"):
            optimizer = tf.train.MomentumOptimizer(
                            learning_rate=self.learning_rate,
                            momentum=0.9)
            grads_and_vars = optimizer.compute_gradients(loss)
            self.train_step = optimizer.apply_gradients(grads_and_vars,
                                                        global_step=self.step)

    def _create_saver(self, var_list):
        return tf.train.Saver(var_list)

    def _create_step(self):
        with tf.variable_scope(self.scope):
            self.step = tf.get_variable(name="step",
                                        initializer=tf.ones(shape=(),
                                                            dtype=tf.int32),
                                        trainable=False)
            self.sess.run(tf.variables_initializer([self.step]))

    def get_step(self):
        return self.sess.run(self.step)

    def _nested_getattr(self, attr):
        # getattr built-in extended with capability
        # of handling nested attributes
        return functools.reduce(getattr, attr.split('.'), self)

    def _smooth_L1(self, x):
        L2 = tf.square(x) / 2
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
            var_list = tf.trainable_variables()
        else:
            var_list = tf.global_variables()

        return list(v for v in var_list if scope in v.name)

    def _number_of_parameters(self, vars_):
        return sum(np.prod(v.get_shape().as_list()) for v in vars_)

    @property
    def n_classes(self):
        return len(self.config.classnames)

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