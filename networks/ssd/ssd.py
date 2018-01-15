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
from ops.postprocessing import non_maximum_supression


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
                    self._create_summaries()
                    self.optimizer_vars = self._get_vars_by_scope("optimizer")
                    self.all_vars += self.optimizer_vars

                self.saver = self._create_saver(self.all_vars)

                if resume:
                    self.load_model()
                else:
                    self._init_vars(self.all_vars)
                    self.vgg_16.load_weights_from_npz(sess=self.sess)

    def _create_placeholders(self):

        with tf.name_scope("inputs"):
            self.is_training = tf.placeholder(dtype=tf.bool, name="is_training")

            if self.mode is TRAIN:
                self.train_tensor_provider = TensorProvider(
                    capacity=20,
                    sess=self.sess,
                    # images, labels, offsets
                    dtypes=(tf.float32,
                            tf.int32,
                            tf.float32),
                    number_of_threads=4)

                self.test_tensor_provider = TensorProvider(
                    capacity=2,
                    sess=self.sess,
                    # images, labels, offsets
                    dtypes=(tf.float32,
                            tf.int32,
                            tf.float32),
                    number_of_threads=2)

                (self.train_images,
                 self.train_labels,
                 self.train_offsets) = self.train_tensor_provider.get_input()
                
                (self.test_images,
                 self.test_labels,
                 self.test_offsets) = self.test_tensor_provider.get_input()

                # choose inputs basen on `is_training` placeholder
                self.images = tf.cond(self.is_training,
                                     lambda: self.train_images,
                                     lambda: self.test_images)

                self.labels = tf.cond(self.is_training,
                                     lambda: self.train_labels,
                                     lambda: self.test_labels)

                self.offsets = tf.cond(self.is_training,
                                     lambda: self.train_offsets,
                                     lambda: self.test_offsets)

            elif self.mode is INFERENCE:
                # just use placeholder as inputs on inference
                self.images = tf.placeholder_with_default(
                    input=tf.zeros((1,) + self.config.input_shape),
                    shape=(None,) + self.config.input_shape,
                    name="test_images")

        self.vgg_16 = VGG_16(self.config.input_shape, self.images)

    def _create_graph(self):

        self.regularizer = slim.l2_regularizer(self.config.weight_decay)
    
        with tf.variable_scope('feedforward_convo'):
            self._create_feedforward_convo()
        with tf.variable_scope('out_convo'):
            self._create_out_convo()

    def _batch_norm(self, net, scope):

        return slim.batch_norm(net,
                               decay=self.config.batch_norm_decay,
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

    def _create_loss(self):

        with tf.name_scope("loss"):
            self.positives, self.negatives = self._positives_and_negatives(
                self.confidences,
                self.labels,
                self.config.neg_pos_ratio
            )
            self.n_positives = tf.reduce_sum(self.positives, 1)
            self.n_negatives = tf.reduce_sum(self.negatives, 1)
            # for summaries
            self.min_positives = tf.reduce_min(self.n_positives)
            self.min_negatives = tf.reduce_min(self.n_negatives)
        
            positives_and_negatives = self.positives + self.negatives
            classification_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.predicted_labels,
                labels=self.labels
            )
            classification_loss *= positives_and_negatives
            classification_loss = tf.reduce_sum(classification_loss, 1)
            total_pos_neg = self.n_positives + self.n_negatives
            classification_loss /= total_pos_neg + 1e-6

            localization_loss = self._smooth_L1(self.predicted_offsets \
                                                - self.offsets)
            localization_loss = tf.reduce_sum(localization_loss, 2)
            localization_loss *= self.positives
            localization_loss = tf.reduce_sum(localization_loss, 1)
            localization_loss /= self.n_positives + 1e-6

            # average over minibatch
            self.classification_loss = tf.reduce_mean(classification_loss, 0)
            self.localization_loss = tf.reduce_mean(localization_loss, 0)

            self.l2_loss = tf.add_n(tf.losses.get_regularization_losses())

            loss = (self.classification_loss +
                    self.localization_loss +
                    self.l2_loss)

            return loss

    def _positives_and_negatives(self, confidences, labels, neg_pos_ratio):

        background_class = 0
        one_hot_labels = tf.one_hot(labels, self.n_classes + 1, axis=2)
        positives = tf.cast(tf.not_equal(labels, background_class), tf.float32)
        n_positives = tf.reduce_sum(positives, 1, keep_dims=True)
        n_negatives = tf.cast(n_positives * neg_pos_ratio, tf.float32)
        true_labels_mask = 1 - one_hot_labels
        non_positive_mask = 1 - positives
        true_confidences = confidences * true_labels_mask
        top_wrong_confidences = tf.reduce_max(true_confidences, axis=2)

        def compute_threshold(inputs):
            confidence = tf.slice(inputs, [0], [self.total_boxes])
            k = tf.slice(inputs, [self.total_boxes], [1])
            k = tf.squeeze(tf.cast(k, tf.int32), 0)
            top, _ = tf.nn.top_k(confidence, k)

            return tf.cond(k > 0,
                           lambda: tf.slice(top, [k-1], [1]),
                           lambda: tf.constant([1.0]))

        # there is a need to concatenate arguments of map,
        # since map doesn't accept multiple arguments
        map_inputs = tf.concat((top_wrong_confidences \
                                * non_positive_mask,
                                n_negatives),
                               axis=1)

        thresholds = tf.map_fn(compute_threshold, map_inputs)

        negatives = tf.cast(tf.greater(top_wrong_confidences \
                                       * non_positive_mask,
                                       thresholds),
                            tf.float32)

        return positives, negatives

    def _create_optimizer(self, loss):

        with tf.variable_scope("optimizer"):

            self.learning_rate = tf.placeholder(dtype=tf.float32,
                                                name="learning_rate")

            optimizer = tf.train.AdamOptimizer(
                            learning_rate=self.learning_rate)
            grads_and_vars = optimizer.compute_gradients(loss)
            self.train_step = optimizer.apply_gradients(grads_and_vars,
                                                        global_step=self.step)

    def _create_summaries(self):

        with tf.name_scope("summaries"):
            tf.summary.scalar("classification_loss", self.classification_loss)
            tf.summary.scalar("localization_loss", self.localization_loss)
            tf.summary.scalar("l2_loss", self.l2_loss)
            tf.summary.scalar("min_positives", self.min_positives)
            tf.summary.scalar("min_negatives", self.min_negatives)
            self.summary = tf.summary.merge_all()

            train_summary_path = os.path.join(self.config.summary_path,
                                              self.scope,
                                              "train")
            self.train_writer = tf.summary.FileWriter(train_summary_path,
                                                      graph=self.sess.graph)
            test_summary_path = os.path.join(self.config.summary_path,
                                             self.scope,
                                             "test")
            self.test_writer = tf.summary.FileWriter(test_summary_path)

    def _create_saver(self, var_list):
        with tf.name_scope("saver"):
            return tf.train.Saver(var_list)

    def _create_step(self):
        with tf.variable_scope("step"):
            self.step = tf.get_variable(name="step",
                                        initializer=tf.ones(shape=(),
                                                            dtype=tf.int32),
                                        trainable=False)
            self._init_vars([self.step])

    def get_step(self):
        return self.sess.run(self.step)

    def _nested_getattr(self, attr):
        # getattr built-in extended with capability
        # of handling nested attributes
        return functools.reduce(getattr, attr.split('.'), self)

    def _smooth_L1(self, x):

        L2 = tf.square(x) / 2
        L1 = tf.abs(x) - 0.5

        cond = tf.cast(tf.less(tf.abs(x), 1.0), tf.float32)
        distance = cond * L2 + (1 - cond) * L1

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

    def _init_vars(self, var_list):
        self.sess.run(tf.variables_initializer(var_list))

    def save_model(self, path=None, sess=None):

        save_dir = path or self.config.model_path
        os.makedirs(save_dir, exist_ok=True)
        self.saver.save(sess or self.sess,
                        os.path.join(save_dir, "model.ckpt"))

    def load_model(self, path=None, sess=None):

        if path is None:
            ckpt = tf.train.get_checkpoint_state(self.config.model_path)
            if ckpt is None:
                raise FileNotFoundError("Can`t load a model. "
                                        "Checkpoint doesn`t exist.")
        restore_path = path or ckpt.model_checkpoint_path
        self.saver.restore(sess or self.sess, restore_path)

    def _confidences_and_corrections(self, feed_dict):
        fetches = [self.confidences, self.predicted_offsets]
        confidences, corrections = self.sess.run(fetches, feed_dict)
        return confidences, corrections

    def fit(self, loader):

        with self.graph.as_default():
            train_data_provider = functools.partial(
                                              loader.train_batch,
                                              batch_size=self.config.batch_size)
            test_data_provider = functools.partial(
                                              loader.test_batch,
                                              batch_size=self.config.batch_size)
            self.train_tensor_provider.set_data_provider(train_data_provider)
            self.test_tensor_provider.set_data_provider(test_data_provider)

            for iteration in range(self.get_step(), self.config.iterations):
                self._train_iteration(iteration)

                if iteration % self.config.log_interval == 0:        
                    self._evaluate(iteration)

                # if iteration % self.config.test_interval == 0:
                #     self._test_iteration(loader, iteration)

                if iteration % self.config.save_interval == 0:
                    self.save_model()

    def _train_iteration(self, iteration):

        fetches = [self.train_step]

        learning_rate = self.config.learning_rate_schedule(iteration)

        feed_dict = {self.learning_rate: learning_rate,
                     self.is_training: True}

        self.sess.run(fetches, feed_dict)

    def _evaluate(self, iteration):
        
        fetches = [self.loss, self.summary]

        # train batch
        feed_dict = {self.is_training: True}

        train_loss, summary = self.sess.run(fetches, feed_dict)
        self.train_writer.add_summary(summary, global_step=self.get_step())

        # test batch
        feed_dict = {self.is_training: False}

        test_loss, summary = self.sess.run(fetches, feed_dict)
        self.test_writer.add_summary(summary, global_step=self.get_step())

        print("Iteration: {}, loss on train: {}, loss on test: {}"
              .format(iteration, train_loss, test_loss))

    def _make_prediction(self, image, loader, filename, save_path):

        feed_dict = {self.test_images: [image], self.is_training: False}
        confidences, corrections = self._confidences_and_corrections(feed_dict)

        image = non_maximum_supression(
                                confidences=confidences.squeeze(),
                                offsets=corrections.squeeze(),
                                default_boxes=loader.default_boxes,
                                image=image,
                                class_mapping=loader.test.class_mapping,
                                nms_threshold=self.config.nms_threshold,
                                max_boxes=self.config.max_boxes,
                                filename=filename)

        if image.bboxes is not None:
            image.plot_image_with_bboxes(save_path,
                                         colormap=loader.test.colormap,
                                         filename=filename)

    def _test_iteration(self, loader, iteration):

        image, filename = loader.single_train_image()
        self._make_prediction(image,
                              loader,
                              filename="{}_{}"
                              .format(filename, iteration),
                              save_path="./predictions/train")

        image, filename = loader.single_test_image()
        self._make_prediction(image,
                              loader,
                              filename="{}_{}"
                              .format(filename, iteration),
                              save_path="./predictions/test")

    