import numpy as np
import pandas as pd
import pytest

from box_arrays import BoundBoxArray


@pytest.fixture
def boundboxes():
    return [(20, 30, 80, 100), (70, 100, 100, 200)]


@pytest.fixture
def centerboxes():
    return [(50, 65, 60, 70), (85, 150, 30, 100)]


@pytest.fixture
def boxes(boundboxes, centerboxes):
    return [bound + center for bound, center in zip(boundboxes, centerboxes)]

@pytest.fixture
def non_overlapping_boundboxes():
    return ([(20, 30, 80, 100), (25, 5, 50, 10)],
            [(110, 120, 120, 140), (200, 200, 300, 300)])

@pytest.fixture
def full_and_quarter_boundboxes():
    return ([(0, 0, 100, 100)],
            [(0, 0, 50, 50)])


@pytest.fixture
def classnames():
    return ["cat", "dog"]


def test_boundbox_creation(boundboxes, centerboxes, boxes, classnames):

    from_boundboxes = BoundBoxArray.from_boundboxes(boundboxes, classnames)

    assert from_boundboxes.shape == (2, 8)
    assert (from_boundboxes.index == classnames).all()

    from_centerboxes = BoundBoxArray.from_centerboxes(centerboxes, classnames)

    assert from_boundboxes.equals(from_centerboxes)

    from_boxes = BoundBoxArray.from_boxes(boxes, classnames)

    assert from_boundboxes.equals(from_boxes)
    assert from_centerboxes.equals(from_boxes)


def test_boundbox_rescale(boxes):

    scale = (3, 4)
    scales = (4, 3) * 4

    boundbox_array = BoundBoxArray.from_boxes(boxes)

    scaled_boxes = [tuple(x / s for x, s in zip(box, scales)) for box in boxes]
    scaled_boundbox_array = BoundBoxArray.from_boxes(scaled_boxes)

    assert scaled_boundbox_array.equals(boundbox_array.rescale(scale))


def test_iou(non_overlapping_boundboxes, full_and_quarter_boundboxes):
    """Test IOU calculation"""

    first, second = non_overlapping_boundboxes
    first = BoundBoxArray.from_boundboxes(first)
    second = BoundBoxArray.from_boundboxes(second)

    assert np.allclose(first.iou(second), np.zeros((2, 2)))
    assert np.allclose(second.iou(first), np.zeros((2, 2)))
    assert np.allclose(first.iou(first), np.eye(2))
    assert np.allclose(second.iou(second), np.eye(2))

    full, quarter = full_and_quarter_boundboxes
    full = BoundBoxArray.from_boundboxes(full)
    quarter = BoundBoxArray.from_boundboxes(quarter)

    assert np.allclose(full.iou(quarter), np.array(0.25))
    assert np.allclose(quarter.iou(full), np.array(0.25))
