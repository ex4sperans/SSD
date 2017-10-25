import os

import numpy as np
import pytest

from containers.image import AnnotatedImage
from containers.box_arrays import BoundBoxArray
from ops.io_ops import load_image, parse_annotation


DIRNAME = os.path.dirname(__file__)
IMAGE = os.path.join(DIRNAME, "mini_voc/test/images", "000010.jpg")
ANNOTATION = os.path.join(DIRNAME, "mini_voc/test/annotations", "000010.xml")


@pytest.fixture
def annotated_image():

    image = load_image(IMAGE)
    annotation = parse_annotation(ANNOTATION)

    return AnnotatedImage(image, annotation)


def test_image_creation(annotated_image):

    assert (annotated_image.bboxes.index == ["horse", "person"]).all()
    assert annotated_image.shape == (480, 354, 3)

    loaded = AnnotatedImage.load(IMAGE, ANNOTATION)
    assert (annotated_image.image == loaded.image).all()
    assert annotated_image.bboxes.equals(loaded.bboxes)


def test_image_resize(annotated_image):

    size = (300, 300)
    resized = annotated_image.resize(size)
    height, width = annotated_image.size
    resized_bboxes = annotated_image.bboxes.rescale((height / 300, width / 300))

    assert resized.size == size
    assert resized.bboxes.equals(resized_bboxes)


def test_image_normalization(annotated_image):

    scale = 255
    normalized = annotated_image.normalize(255)

    assert ((0 <= normalized.image) & (normalized.image <= 1)).all()
    assert normalized.bboxes.equals(annotated_image.bboxes)


def test_bboxes_normalization(annotated_image):

    normalized = annotated_image.normalize_bboxes()

    assert (normalized.bboxes.x_min >= 0).all()
    assert (normalized.bboxes.y_min >= 0).all()
    assert (normalized.bboxes.x_max <= 1).all()
    assert (normalized.bboxes.y_max <= 1).all()

    original = normalized.recover_bboxes()

    assert np.allclose(original.bboxes, annotated_image.bboxes)


def test_labels_and_offsets():

    default_boxes = BoundBoxArray.from_boundboxes([(0, 0, 0.5, 0.5),
                                                   (0, 0, 1.0, 1.0),
                                                   (0.45, 0.45, 0.9, 0.9)])

    bboxes = BoundBoxArray.from_boundboxes([(0, 0, 150, 150),
                                            (0, 0, 120, 120),
                                            (150, 150, 300, 300)],
                                           classnames=["cat", "pig", "dog"])
    class_mapping = dict(cat=1, pig=2, dog=3)
    threshold = 0.5

    image = AnnotatedImage(np.ones((300, 300, 3)), bboxes)
    image = image.normalize_bboxes()

    labels, offsets = image.labels_and_offsets(default_boxes,
                                               threshold,
                                               class_mapping)

    # cat matched to first, dog matched to third
    # pig wasn't matched since cat has higher IOU
    assert (labels == [1, 0, 3]).all()
    # cat matched perfectly, second default box
    # wasn't matched
    assert (offsets[[0, 1]] == [0, 0, 0, 0]).all()
