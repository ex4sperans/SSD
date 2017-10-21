import numpy as np
import pandas as pd
import pytest

from box_arrays import CenterBoxArray, BoundBoxArray


@pytest.fixture
def boundboxes():
    return [(20.0, 30.0, 80.0, 100.0), (70.0, 100.0, 100.0, 200.0)]

@pytest.fixture
def centerboxes():
    return [(50.0, 65.0, 60.0, 70.0), (85.0, 150.0, 30.0, 100.0)]


@pytest.fixture
def classnames():
    return ["cat", "dog"]


def test_boundbox_array_creation(boundboxes, classnames):
    """Test functionality of BoundBoxArray creation"""

    boundbox_array = BoundBoxArray.from_boxes(boundboxes, classnames)

    assert boundbox_array.shape == (2, 4)
    assert (boundbox_array.index == classnames).all()

    # without classnames
    boundbox_array = BoundBoxArray.from_boxes(boundboxes)

    assert boundbox_array.shape == (2, 4)
    assert (boundbox_array.index == (0, 1)).all()


def test_center_array_creation(centerboxes, classnames):
    """Test functionality of CenterBoxArray creation"""

    centerbox_array = CenterBoxArray.from_boxes(centerboxes, classnames)

    assert centerbox_array.shape == (2, 4)
    assert (centerbox_array.index == classnames).all()

    # without classnames
    centerbox_array = CenterBoxArray.from_boxes(centerboxes)

    assert centerbox_array.shape == (2, 4)
    assert (centerbox_array.index == (0, 1)).all()


def test_box_conversion(boundboxes, centerboxes, classnames):
    """Test functionality of conversion between arrays"""

    boundbox_array = BoundBoxArray.from_boxes(boundboxes, classnames)
    centerbox_array = CenterBoxArray.from_boxes(centerboxes, classnames)

    assert boundbox_array.equals(centerbox_array.as_boundbox_array())
    assert centerbox_array.equals(boundbox_array.as_centerbox_array())


def test_box_rescale(boundboxes, centerboxes):
    """Test functionality of boxes scaling"""

    scale = (3, 4)
    scales = (4, 3, 4, 3)

    boundbox_array = BoundBoxArray.from_boxes(boundboxes)
    centerbox_array = CenterBoxArray.from_boxes(centerboxes)


    scaled_boundboxes = [tuple(x / s for x, s in zip(box, scales))
                         for box in boundboxes]
    scaled_boundbox_array = BoundBoxArray.from_boxes(scaled_boundboxes)

    scaled_centerboxes = [tuple(x / s for x, s in zip(box, scales))
                          for box in centerboxes]
    scaled_centerbox_array = CenterBoxArray.from_boxes(scaled_centerboxes)

    assert scaled_boundbox_array.equals(boundbox_array.rescale(scale))
    assert scaled_centerbox_array.equals(centerbox_array.rescale(scale))

    assert np.allclose(boundbox_array.rescale(scale).as_matrix(),
        centerbox_array.rescale(scale).as_boundbox_array().as_matrix())
