import pytest

from default_boxes import get_default_boxes, get_default_box
from box_arrays import BoundBoxArray


def test_get_default_box():

    i = 3
    j = 2
    scale = 1
    box_ratio = 1
    width = 4
    height = 3

    default_box = get_default_box(i=i,
                                  j=j,
                                  scale=scale,
                                  box_ratio=box_ratio,
                                  width=width,
                                  height=height)

    center_x, center_y, box_width, box_height = default_box

    assert center_x == (j + 1/2) / width
    assert center_y == (i + 1/2) / height
    assert box_width == 1
    assert box_height == 1


def test_get_default_boxes():

    out_shapes = [(3, 3, 64), (2, 3, 32)]
    box_ratios = [(1, 2), (1, 1/2)]

    default_boxes = get_default_boxes(out_shapes, box_ratios)

    assert isinstance(default_boxes, BoundBoxArray)
    assert len(default_boxes) == 3 * 3 * 2 + 2 * 3 * 2
