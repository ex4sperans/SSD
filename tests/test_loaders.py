import os

import numpy as np

from loaders.voc_loader import VOCLoader
from ops.default_boxes import get_default_boxes

DIRNAME = os.path.dirname(__file__)


def test_voc_loader():

    TRAIN_IMAGES = os.path.join(DIRNAME, "mini_voc/train/images")
    TRAIN_ANNOTATIONS = os.path.join(DIRNAME, "mini_voc/train/annotations")
    TEST_IMAGES = os.path.join(DIRNAME, "mini_voc/test/images")
    TEST_ANNOTATIONS = os.path.join(DIRNAME, "mini_voc/test/annotations")

    default_boxes = get_default_boxes([(9, 9, 32), (5, 5, 32)], 2 * [(1, 3, 1/3)])
    n_default_boxes = len(default_boxes)

    loader = VOCLoader(TRAIN_IMAGES, TRAIN_ANNOTATIONS,
                       TEST_IMAGES, TEST_ANNOTATIONS,
                       default_boxes=default_boxes,
                       resize_to=(300, 300),
                       matching_threshold=0.45,
                       max_samples=3)

    for images, labels, offsets in loader.train_iterator(batch_size=2,
                                                         iterations=10):
        assert images.shape == (2, 300, 300, 3)
        assert images.dtype == np.float32
        assert labels.shape == (2, n_default_boxes)
        assert labels.dtype == np.int32
        assert offsets.shape == (2, n_default_boxes, 4)
        assert offsets.dtype == np.float32
