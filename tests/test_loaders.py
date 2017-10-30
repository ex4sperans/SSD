import os

import numpy as np
import pytest

from loaders.voc_loader import VOCLoader
from datasets.voc_dataset import VOCDataset
from ops.default_boxes import get_default_boxes
from ops.io_ops import find_files

DIRNAME = os.path.dirname(__file__)


@pytest.fixture(scope="module")
def voc_loader():

    TRAIN_IMAGES = os.path.join(DIRNAME, "mini_voc/train/images")
    TRAIN_ANNOTATIONS = os.path.join(DIRNAME, "mini_voc/train/annotations")
    TEST_IMAGES = os.path.join(DIRNAME, "mini_voc/test/images")
    TEST_ANNOTATIONS = os.path.join(DIRNAME, "mini_voc/test/annotations")

    default_boxes = get_default_boxes([(9, 9, 32), (5, 5, 32)],
                                       2 * [(1, 3, 1/3)])

    return VOCLoader(TRAIN_IMAGES, TRAIN_ANNOTATIONS,
                     TEST_IMAGES, TEST_ANNOTATIONS,
                     train_transform=lambda image: (image
                                                    .normalize(255)
                                                    .normalize_bboxes()),
                     test_transform=lambda image: (image
                                                   .normalize(255)
                                                   .normalize_bboxes()),
                     default_boxes=default_boxes,
                     resize_to=(300, 300),
                     matching_threshold=0.45,
                     max_samples=3)


def test_voc_loader_train_iterator(voc_loader):

    for (images,
         labels,
         offsets) in voc_loader.random_train_iterator(batch_size=2,
                                                      iterations=10):
        assert images.shape == (2, 300, 300, 3)
        assert images.dtype == np.float32
        assert labels.shape == (2, len(voc_loader.default_boxes))
        assert labels.dtype == np.int32
        assert offsets.shape == (2,  len(voc_loader.default_boxes), 4)
        assert offsets.dtype == np.float32


def test_voc_loader_process_image(voc_loader):

    for image in voc_loader.train:

        classnames = set(image.bboxes.index)
        (image,
         labels,
         offsets) = voc_loader.process_image(image, voc_loader.train_transform)
        expected_labels = set(VOCDataset.class_mapping[n] for n in classnames)

        assert expected_labels.union({voc_loader.background}) == set(labels)


def test_voc_loader_single_train_image(voc_loader):

    image, filename = voc_loader.single_train_image()
    assert (image >= 0).all and (image <= 1).all()
