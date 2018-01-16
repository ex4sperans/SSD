"""Various augmentation ops"""

import numpy as np

from containers.box_arrays import BoundBoxArray


def hflip(image):
    """Flips an instance of AnnotatedImage"""

    flipped_image = np.fliplr(image.image)
    normalized_image = image.normalize_bboxes()

    bboxes = normalized_image.bboxes.boundboxes
    flipped_bboxes = (1 - bboxes.x_max, 1 - bboxes.y_max,
                      1 - bboxes.x_min, 1 - bboxes.y_min)
    flipped_bboxes = np.array(flipped_bboxes, dtype=np.float32).T
    flipped_bboxes = BoundBoxArray.from_boundboxes(
        flipped_bboxes,
        classnames=bboxes.classnames
    )
    flipped = image.__class__(flipped_image,
                              flipped_bboxes,
                              filename=image.filename,
                              bboxes_normalized=True)
    return flipped