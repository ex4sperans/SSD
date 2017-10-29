import numpy as np

from ops.box_ops import calculate_offsets, apply_offsets
from containers.box_arrays import BoundBoxArray


def test_offsets():

    original = BoundBoxArray.from_boundboxes([(0, 0, 0.5, 0.5),
                                              (0.3, 0.4, 0.6, 0.9)])
    default_boxes = BoundBoxArray.from_boundboxes([(0, 0, 0.5, 0.4),
                                                   (0.2, 0.4, 0.6, 0.9)])

    # get offsets 
    offsets = calculate_offsets(default_boxes, original)
    offsets = BoundBoxArray.from_centerboxes(offsets)

    # apply them to get original bboxes
    recovered = apply_offsets(default_boxes, offsets)
    recovered = BoundBoxArray.from_centerboxes(recovered)

    assert np.allclose(original, recovered)