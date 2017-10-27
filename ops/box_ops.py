import numpy as np


def _intersection(first, second):
    """Given two boundbox arrays, computes pairwise intersections 
    between bouding boxes"""

    intersection_w = (np.add.outer(first.width, second.width) -
                      (np.maximum.outer(first.x_max, second.x_max) -
                       np.minimum.outer(first.x_min, second.x_min))
                     )
    intersection_h = (np.add.outer(first.height, second.height) -
                      (np.maximum.outer(first.y_max, second.y_max) -
                       np.minimum.outer(first.y_min, second.y_min))
                     )

    return np.maximum(intersection_w, 0) * np.maximum(intersection_h, 0)


def _union(first, second, intersection):
    """Given two boundbox arrays, and precomputed intersection,
    computes pairwise union between bounding boxes"""

    return (np.add.outer(first.width * first.height,
                         second.width * second.height) -
            intersection)


def iou(first, second):
    """Given two boundbox arrays, computes pairwise IOU between
    bounding boxes"""

    intersection = _intersection(first, second)
    union = _union(first, second, intersection)

    return np.maximum(intersection / (union + 1e-10), 0)


def calculate_offsets(default, matched):
    """Computes offsets between given default boxes and matched boxes"""

    return np.transpose(((matched.x_center - default.x_center) / default.width,
                         (matched.y_center - default.y_center) / default.height,
                         np.log(matched.width / default.width),
                         np.log(matched.height / default.height)))


def apply_offsets(default, offsets):
    """Applies offsets for given default boxes"""

    return np.transpose((default.x_center + offsets.x_center * default.width,
                         default.y_center + offsets.y_center * default.height,
                         default.width * np.exp(offsets.width),
                         default.height * np.exp(offsets.height)))