"""Postprocess detection results"""

import numpy as np

from containers.box_arrays import BoundBoxArray
from containers.image import AnnotatedImage
from ops.box_ops import apply_offsets
from ops.misc import reverse_dict


def non_maximum_supression(confidences, offsets, default_boxes,
                           class_mapping, image, nms_threshold,
                           filename, max_boxes, clip=True):
    """Given confidences, default boxes and offsets,
    outputs top confident boxes and theirs classes

    Args:
        confidences: np.ndarray of shape (N_BOUNDING_BOXES, N_CLASSES + 1)
            with class probabilities for every default box
        offsets: np.ndarray of shape (N_BOUNDING_BOXES, 4)
            with corrections to default boxes
        default_boxes: BoundBoxArray with default boxes
        classnames: a mapping from classnames to labels
        image: np.ndarray, image
        nms_threshold: float, used during non-maximum-supression
            to decide if two boxes correspond to one object
        max_boxes: int, maximum number of boxes
        filename: str, filename
        clip: bool, whether to clip offsets to (0, 1)

    Returns:
        AnnotatedImage with bounding boxes
    """

    reverse_mapping = reverse_dict(class_mapping)
    background = 0

    labels = confidences.argmax(axis=1)
    top_confidences = confidences.max(axis=1)

    # choose only non-background boxes
    choices = np.not_equal(labels, background)

    top_confidences = np.compress(choices, top_confidences)
    labels = np.compress(choices, labels)
    offsets = np.compress(choices, offsets, axis=0)
    default_boxes = default_boxes[choices]

    # sort confidences
    sorted_confidence_idx = np.flip(np.argsort(top_confidences), 0)

    top_confidences = np.take(top_confidences, sorted_confidence_idx)
    labels = np.take(labels, sorted_confidence_idx)
    offsets = np.take(offsets, sorted_confidence_idx, axis=0)
    default_boxes = BoundBoxArray.from_boxes(
                        default_boxes.iloc[sorted_confidence_idx].as_matrix())

    # apply offsets and construct BoundBoxArray
    if clip:
        offsets = np.clip(offsets, 0, 1)
    offsets = BoundBoxArray.from_centerboxes(offsets)
    corrected_boxes = apply_offsets(default_boxes, offsets)
    corrected_boxes = BoundBoxArray.from_centerboxes(corrected_boxes)

    top_boxes = None

    for (confidence, label, box) in zip(
            top_confidences, labels, corrected_boxes.centerboxes.as_matrix()):

        classname = reverse_mapping[label]
        box = BoundBoxArray.from_centerboxes([box], [classname])

        if top_boxes is None:
            top_boxes = box
            continue

        if len(top_boxes) >= max_boxes:
            break

        non_matching = (top_boxes.iou(box) < nms_threshold).squeeze()
        matching = np.logical_not(non_matching)

        box_class = [top_boxes_classname == classname
                     for top_boxes_classname
                     in top_boxes.classnames]

        matching_box_class = np.logical_and(matching, box_class)
        # add box if it either doesn't match with any of the already
        # selected boxes or all matches are with the other class
        if non_matching.all() or not matching_box_class.any():
            top_boxes = top_boxes.append(box)

    return AnnotatedImage(image, top_boxes, filename, bboxes_normalized=True)
