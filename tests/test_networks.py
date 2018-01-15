import pytest
import tensorflow as tf
import numpy as np

from datasets.voc_dataset import VOCDataset
from networks.vgg.vgg import VGG_16
from networks.ssd.ssd import SSD
from networks.ssd.ssd import TRAIN, INFERENCE
from networks.ssd.ssd import OutConvoLayer

np.random.seed(42)

### VGG16 ###

def test_vgg_16():

    with tf.Graph().as_default():

        inputs = tf.placeholder(dtype=tf.float32, shape=(None, 300, 300, 3))

        vgg_16 = VGG_16((300, 300, 3), inputs)

        assert vgg_16.convo_output.get_shape().as_list() == [None, 9, 9, 512]

        with tf.Session() as sess:

            vgg_16.load_weights_from_npz(sess)


### SSD ###

@pytest.fixture(scope="module")
def ssd_config():

    class Config:

        scope = "SSD"
        input_shape = (300, 300, 3)
        weight_decay = 0.0005
        batch_norm_decay = 0.995
        tune_base = True
        neg_pos_ratio = 3

        out_layers = [OutConvoLayer(name="out_convo4_3",
                                    parent="vgg_16.conv4_3",
                                    kernel_size=(3, 3),
                                    box_ratios=(1, 2, 1/2)),
                      OutConvoLayer(name="out_convo7",
                                    parent="conv7",
                                    kernel_size=(3, 3),
                                    box_ratios=(1, 2, 1/2))]

        classnames = VOCDataset.classnames
        background_class = VOCDataset.background
    
    return Config


@pytest.fixture(scope="module")
def ssd(ssd_config):

    return SSD(ssd_config(), mode=INFERENCE, resume=False)


def test_ssd_construction(ssd):

    assert ssd.conv11_2.get_shape().as_list() == [None, 1, 1, 256]


def test_ssd_positives_and_negatives(ssd):

    batch_size = 4
    neg_pos_ratio = 3
    background_class = ssd.config.background_class

    with ssd.graph.as_default():

        confidences = np.random.normal(
            size=(batch_size, ssd.total_boxes, ssd.n_classes + 1)
        ).astype(np.float32)
        exps = np.exp(confidences)
        confidences = exps / exps.sum(2, keepdims=True)

        labels = np.random.randint(
            0, ssd.n_classes + 1, size=(batch_size, ssd.total_boxes)
        )
        # make most of the entries of labels to be zero 
        # in order to simulate realistic setting
        background_mask = (np.random.uniform(size=labels.shape) > 0.95)
        labels *= background_mask
        one_hot_labels = tf.one_hot(labels, ssd.n_classes + 1, axis=2)

        positives, negatives = ssd._positives_and_negatives(
            confidences, labels, neg_pos_ratio
        )

        positives, negatives, one_hot_labels = ssd.sess.run(
            [positives, negatives, one_hot_labels]
        )

    # ensure that there are approximately
    # 3x more negatives than positives
    assert np.allclose(
        positives.sum(axis=1) * neg_pos_ratio, negatives.sum(axis=1),
        atol=2
    )
    # check positives
    assert np.array_equal(positives, (labels != background_class))
    # check negatives
    n_negatives = (positives.sum(1) * neg_pos_ratio).astype(np.int32)
    non_positives = 1 - positives
    wrong_class = 1 - one_hot_labels
    wrong_confidences = confidences * wrong_class
    top_wrong_confidences = wrong_confidences.max(axis=2)
    threshold = np.sort(top_wrong_confidences * non_positives, axis=1)
    threshold = np.flip(threshold, axis=1)
    threshold = threshold[range(batch_size), n_negatives - 1][:, np.newaxis]
    new_negatives = (top_wrong_confidences * non_positives > threshold)
    new_negatives = new_negatives.astype(np.float32)
    assert np.array_equal(negatives, new_negatives)
