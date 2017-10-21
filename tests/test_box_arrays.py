import numpy as np
import pandas as pd
import pytest

from box_arrays import BoundBoxArray


@pytest.fixture
def boundboxes():
    return [(20.0, 30.0, 80.0, 100.0), (70.0, 100.0, 100.0, 200.0)]


@pytest.fixture
def centerboxes():
    return [(50.0, 65.0, 60.0, 70.0), (85.0, 150.0, 30.0, 100.0)]


@pytest.fixture
def boxes(boundboxes, centerboxes):
    return [bound + center for bound, center in zip(boundboxes, centerboxes)]


@pytest.fixture
def classnames():
    return ["cat", "dog"]


def test_boundbox_creation(boundboxes, centerboxes, boxes, classnames):
    """Test creation of BoundBoxArray"""

    from_boundboxes = BoundBoxArray.from_boundboxes(boundboxes, classnames)

    assert from_boundboxes.shape == (2, 8)
    assert (from_boundboxes.index == classnames).all()

    from_centerboxes = BoundBoxArray.from_centerboxes(centerboxes, classnames)

    assert from_boundboxes.equals(from_centerboxes)

    from_boxes = BoundBoxArray.from_boxes(boxes, classnames)

    assert from_boundboxes.equals(from_boxes)
    assert from_centerboxes.equals(from_boxes)


def test_box_rescale(boxes):
    """Test functionality of boxes scaling"""

    scale = (3, 4)
    scales = (4, 3) * 4

    boundbox_array = BoundBoxArray.from_boxes(boxes)

    scaled_boxes = [tuple(x / s for x, s in zip(box, scales)) for box in boxes]
    scaled_boundbox_array = BoundBoxArray.from_boxes(scaled_boxes)

    assert scaled_boundbox_array.equals(boundbox_array.rescale(scale))
