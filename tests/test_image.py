import os

import pytest

from image import AnnotatedImage
from io_ops import load_image, parse_annotation


DIRNAME = os.path.dirname(__file__)
IMAGE = os.path.join(DIRNAME, "data", "000010.jpg")
ANNOTATION = os.path.join(DIRNAME, "data", "000010.xml")


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
    assert (annotated_image.bboxes.equals(loaded.bboxes))


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
