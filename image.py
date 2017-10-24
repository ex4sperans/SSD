import numpy as np
from scipy.misc import imresize

from misc import height_and_width
from io_ops import parse_annotation, load_image
from plotting import plot_image, plot_with_bboxes
from box_ops import calculate_offsets


class AnnotatedImage: 

    @classmethod
    def load(cls, image, annotation):
        """Initialize AnnotatedImage from .jpg image and .xml annotation"""

        return cls(image=load_image(image),
                   bboxes=parse_annotation(annotation))

    def __init__(self, image, bboxes, filename=None):
        """Initialize AnnotatedImage object"""

        self._image = image
        self._bboxes = bboxes
        self._filename = filename

    @property
    def image(self):
        return self._image

    @property
    def bboxes(self):
        return self._bboxes

    @property
    def filename(self):
        return self._filename

    @property
    def shape(self):
        return self._image.shape

    @property
    def size(self):
        return height_and_width(self.shape)

    def normalize(self, scale):
        """Normalize image according to `scale`"""
        normalized = self._image / scale

        return AnnotatedImage(normalized, self._bboxes, self._filename)

    def normalize_bboxes(self):
        """Normalize bboxes to be in range of (0, 1) for both axes"""

        return AnnotatedImage(self.image,
                              self.bboxes.rescale(self.size),
                              self.filename)

    def resize(self, size):
        """Resize image and bboxes according to `size`"""
        new_height, new_width = size
        height, width = height_and_width(self.shape)
        scale = (height / new_height, width / new_width)

        return AnnotatedImage(imresize(self.image, size),
                              self.bboxes.rescale(scale),
                              self.filename)

    def labels_and_offsets(self, default_boxes, threshold, class_mapping):
        """Performs matching step."""

        iou = default_boxes.iou(self.bboxes)
        # ensure that each box is matched with
        # single ground-truth box with top IOU
        top_match = iou.max(axis=1, keepdims=True)
        iou *= (iou == top_match)
        matched = iou > threshold

        # labels
        classmask = np.array([class_mapping[classname]
                              for classname in self.bboxes.index],
                             dtype=np.int32)
        labels = (matched * classmask).sum(axis=1, dtype=np.int32)

        # offsets
        default_matched = matched.any(axis=1)
        box_matched = matched.any(axis=0)
        compressed_offsets = calculate_offsets(default_boxes[default_matched],
                                               self.bboxes[box_matched])
        offsets = np.zeros((default_matched.size, 4), dtype=np.float32)
        offsets[default_matched] = compressed_offsets

        return labels, offsets

    def plot(self, save_path, filename=None):
        """Plots and save image"""

        filename = filename or self.filename
        if not filename:
            raise ValueError("`filename` should be specified either on image"
                             " creation or when calling this function.")

        plot_image(self.image, save_path, filename)

    def plot_bboxes(self, save_path, filename=None):
        """Plots and save image with bounding boxes"""

        filename = filename or self.filename
        if not filename:
            raise ValueError("`filename` should be specified either on image"
                             " creation or when calling this function.")

        plot_with_bboxes(self.image, self.bboxes, save_path, filename)