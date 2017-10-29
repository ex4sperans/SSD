import numpy as np

from ops.postprocessing import non_maximum_supression
from containers.image import AnnotatedImage
from ops.default_boxes import get_default_boxes


def test_non_maximum_supression():

    default_boxes = get_default_boxes([(9, 9, 32), (5, 5, 32)],
                                      [(1, 1/3, 3), (1, 1/3, 3)])
    n_boxes = len(default_boxes)

    class_mapping = dict(cat=1,
                         dog=2,
                         cow=3)

    confidences = np.array([(1, 0, 0, 0)] * n_boxes, dtype=np.float32)
    # default box #6 should be associated with class `dog`
    confidences[5] = (0.0, 0.05, 0.9, 0.05)

    # default box #162 should be associated with class `cat`
    # and box #161 should NOT, since it has lower confidence
    # and these two boxes have big overlap
    confidences[160] = (0.0, 0.75, 0, 0.25)
    confidences[161] = (0.0, 0.8, 0.1, 0.11)

    # default box #201 should be associated with class `cat`
    confidences[200] = (0.0, 0.70, 0, 0.30)

    offsets = np.zeros((n_boxes, 4), dtype=np.float32)
    image = np.zeros((300, 300, 3))
    nms_threshold = 0.5
    filename = "foo"
    max_boxes = 10

    annotated_image = non_maximum_supression(confidences, offsets,
                                             default_boxes, class_mapping,
                                             image, nms_threshold,
                                             filename, max_boxes)

    bboxes = annotated_image.bboxes

    assert isinstance(annotated_image, AnnotatedImage)

    assert len(bboxes) == 3
    assert set(bboxes.classnames) == {"cat", "dog"}
    assert np.allclose(bboxes.loc["dog"], default_boxes.iloc[5])
    assert np.allclose(bboxes.loc["cat"],
                       default_boxes.iloc[[161, 200]])

    assert annotated_image.filename == filename