"""Various augmentation ops"""

import numpy as np
from scipy.misc import imresize

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


def crop(image, selection):
    """Crops selection (which is a 4-length tuple
    with normalized (to (0, 1) interval)
    (x_min, y_min, x_max, y_max) coordinates) from image"""

    x_min, y_min, x_max, y_max = selection
    height, width = image.size

    x_min_int = int(x_min * width)
    y_min_int = int(y_min * height)
    x_max_int = int(x_max * width)
    y_max_int = int(y_max * height)

    cropped_image = image.image[y_min_int:y_max_int, x_min_int:x_max_int]
    cropped_image = imresize(cropped_image, image.size)

    scale_x = (x_max - x_min)
    scale_y = (y_max - y_min)
    normalized_bboxes = image.normalize_bboxes().bboxes
    # discard all bboxes that lie outside the crop
    valid_bboxes = (
        (normalized_bboxes.centerboxes.x_center > x_min) &
        (normalized_bboxes.centerboxes.x_center < x_max) &
        (normalized_bboxes.centerboxes.y_center < y_max) &
        (normalized_bboxes.centerboxes.y_center > y_min)
    )

    # return original image if there are no bboxes in the selection
    if not valid_bboxes.any():
        return image

    normalized_bboxes = normalized_bboxes[valid_bboxes]
    cropped_bboxes = normalized_bboxes.clip(
        vertical_clip_value=(y_min, y_max),
        horizontal_clip_value=(x_min, x_max)
    )
    shifted_bboxes = cropped_bboxes.boundboxes - [x_min, y_min, x_min, y_min]
    shifted_bboxes = BoundBoxArray.from_boundboxes(
        shifted_bboxes,
        classnames=cropped_bboxes.classnames
    )
    resized_bboxes = shifted_bboxes.rescale((scale_y, scale_x)).clip()
    
    cropped = image.__class__(cropped_image,
                              resized_bboxes,
                              filename=image.filename,
                              bboxes_normalized=True)
    
    return cropped
