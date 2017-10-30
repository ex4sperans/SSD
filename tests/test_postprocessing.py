import numpy as np

from ops.postprocessing import non_maximum_supression
from containers.image import AnnotatedImage
from containers.box_arrays import BoundBoxArray
from ops.default_boxes import get_default_boxes


def test_non_maximum_supression():

    default_boxes = get_default_boxes([(9, 9, 32), (3, 3, 32)],
                                      [(1, 1/3, 3), (1, 1/3, 3)])
    n_boxes = len(default_boxes)

    class_mapping = dict(cat=1,
                         dog=2,
                         cow=3,
                         table=4)

    confidences = np.array([(1, 0, 0, 0)] * n_boxes, dtype=np.float32)
    # default box #6 should be associated with class `dog`
    confidences[5] = (0.0, 0.05, 0.9, 0.05)
    # default box #101 should be associated with class `cow`
    confidences[100] = (0.0, 0.05, 0.0, 0.95)
    # default box #245 should be associated with class `cat`
    # and box #244 and #246 should NOT since they
    # have lower confidence and these boxes have big overlap
    confidences[243] = (0.0, 0.75, 0, 0.25)
    confidences[244] = (0.0, 0.9, 0.1, 0.0)
    confidences[245] = (0.0, 0.8, 0.1, 0.1)
    # default box #201 should be associated with class `cat`
    confidences[200] = (0.0, 0.70, 0, 0.30)

    offsets = np.zeros((n_boxes, 4), dtype=np.float32)
    offsets[100] = (0.5, 0, np.log(0.5), np.log(0.1))

    image = np.zeros((300, 300, 3))
    nms_threshold = 0.5
    filename = "foo"
    max_boxes = 10

    annotated_image = non_maximum_supression(confidences, offsets,
                                             default_boxes, class_mapping,
                                             image, nms_threshold,
                                             filename, max_boxes, clip=False)

    bboxes = annotated_image.bboxes

    assert isinstance(annotated_image, AnnotatedImage)

    assert len(bboxes) == 4
    assert set(bboxes.classnames) == {"cat", "dog", "cow"}
    assert np.allclose(bboxes.loc["dog"], default_boxes.iloc[5])
    assert np.allclose(bboxes.loc["cat"],
                       default_boxes.iloc[[244, 200]])

    cow_box = default_boxes.centerboxes.iloc[100]
    cow_box.x_center += 0.5 * cow_box.width
    cow_box.width *= 0.5
    cow_box.height *= 0.1
    cow_box = BoundBoxArray.from_centerboxes([cow_box.as_matrix()])
    assert np.allclose(bboxes.loc["cow"], cow_box)

    assert annotated_image.filename == filename
