import numpy as np
import pandas as pd
import pytest

from box_arrays import CenterBoxArray, BoundBoxArray

@pytest.fixture
def initial_boundboxes():
    return [(20.0, 30.0, 80.0, 100.0), (70.0, 100.0, 100.0, 200.0)]

@pytest.fixture
def initial_centerboxes():
    return [(50.0, 65.0, 60.0, 70.0), (85.0, 150.0, 30.0, 100.0)]


@pytest.fixture
def classnames():
    return ["cat", "dog"]


def test_boundbox_array_creation(initial_boundboxes, classnames):
    """Test functionality of BoundBoxArray creation"""

    boundbox_array = BoundBoxArray.from_boxes(initial_boundboxes, classnames)

    assert boundbox_array.shape == (2, 4)
    assert (boundbox_array.index == classnames).all()

    # without classnames
    boundbox_array = BoundBoxArray.from_boxes(initial_boundboxes)

    assert boundbox_array.shape == (2, 4)
    assert (boundbox_array.index == (0, 1)).all()


def test_center_array_creation(initial_centerboxes, classnames):
    """Test functionality of CenterBoxArray creation"""

    centerbox_array = CenterBoxArray.from_boxes(initial_centerboxes, classnames)

    assert centerbox_array.shape == (2, 4)
    assert (centerbox_array.index == classnames).all()

    # without classnames
    centerbox_array = CenterBoxArray.from_boxes(initial_centerboxes)

    assert centerbox_array.shape == (2, 4)
    assert (centerbox_array.index == (0, 1)).all()


def test_box_conversion(initial_boundboxes, initial_centerboxes, classnames):
    """Test functionality of conversion between arrays"""

    boundbox_array = BoundBoxArray.from_boxes(initial_boundboxes, classnames)
    centerbox_array = CenterBoxArray.from_boxes(initial_centerboxes, classnames)

    assert boundbox_array.equals(centerbox_array.as_boundbox_array())
    assert centerbox_array.equals(boundbox_array.as_centerbox_array())