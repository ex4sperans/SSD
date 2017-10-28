import os

import numpy as np

from loaders.voc_loader import VOCLoader
from datasets.voc_dataset import VOCDataset
from ops.default_boxes import get_default_boxes
from ops.io_ops import find_files

DIRNAME = os.path.dirname(__file__)


def test_voc_loader():

    TRAIN_IMAGES = os.path.join(DIRNAME, "mini_voc/train/images")
    TRAIN_ANNOTATIONS = os.path.join(DIRNAME, "mini_voc/train/annotations")
    TEST_IMAGES = os.path.join(DIRNAME, "mini_voc/test/images")
    TEST_ANNOTATIONS = os.path.join(DIRNAME, "mini_voc/test/annotations")

    default_boxes = get_default_boxes([(9, 9, 32), (5, 5, 32)],
                                       2 * [(1, 3, 1/3)])
    n_default_boxes = len(default_boxes)

    loader = VOCLoader(TRAIN_IMAGES, TRAIN_ANNOTATIONS,
                       TEST_IMAGES, TEST_ANNOTATIONS,
                       default_boxes=default_boxes,
                       resize_to=(300, 300),
                       matching_threshold=0.45,
                       max_samples=3)

    for images, labels, offsets in loader.random_train_iterator(batch_size=2,
                                                                iterations=10):
        assert images.shape == (2, 300, 300, 3)
        assert images.dtype == np.float32
        assert labels.shape == (2, n_default_boxes)
        assert labels.dtype == np.int32
        assert offsets.shape == (2, n_default_boxes, 4)
        assert offsets.dtype == np.float32


    for image in loader._train:

        classnames = set(image.bboxes.index)
        image, labels, offsets = loader.process_image(image)
        expected_labels = (VOCDataset.class_mapping[n] for n in classnames)

        # 0 stands for background class
        assert set(expected_labels).union({0}) == set(labels)