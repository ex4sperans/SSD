import numpy as np
from scipy.misc import imresize

from ops.misc import height_and_width, file_name, reverse_dict
from ops.io_ops import parse_annotation, load_image
from ops.plotting import plot_image, plot_with_bboxes
from ops.box_ops import calculate_offsets
from containers.box_arrays import BoundBoxArray


class AnnotatedImage:

    @classmethod
    def load(cls, image, annotation):
        """Initialize AnnotatedImage from .jpg image and .xml annotation"""

        return cls(image=load_image(image),
                   bboxes=parse_annotation(annotation),
                   filename=file_name(image))

    def __init__(self, image, bboxes, filename=None,
                 bboxes_normalization_scale=(1, 1)):
        """Initialize AnnotatedImage object"""

        self._image = image
        self._bboxes = bboxes
        self._filename = filename
        # keep scale to be able to convert
        # normalized bboxes to image scale
        self._bboxes_normalization_scale = bboxes_normalization_scale

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
                              self.filename,
                              self.size)

    def recover_bboxes(self):
        """Recover bboxes to represent size in pixels"""

        height, width = self._bboxes_normalization_scale
        scale = (1 / height, 1 / width)

        return AnnotatedImage(self.image,
                              self.bboxes.rescale(scale),
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
        top_default_match = iou.max(axis=1, keepdims=True)
        iou *= (iou == top_default_match)
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
        """Plot and save image"""

        filename = filename or self.filename
        if not filename:
            raise ValueError("`filename` should be specified either on image"
                             " creation or when calling this function.")

        plot_image(self.image, save_path, filename)

    def plot_image_with_bboxes(self, save_path, colormap, filename=None):
        """Plot and save image with bounding boxes"""

        filename = filename or self.filename
        if not filename:
            raise ValueError("`filename` should be specified either on image"
                             " creation or when calling this function.")

        recovered = self.recover_bboxes()

        plot_with_bboxes(recovered.image, recovered.bboxes,
                         colormap, save_path, filename)

    def plot_matching_bboxes(self, save_path, default_boxes, threshold,
                             class_mapping, colormap, filename=None):
        """Plot and save image with matching bounding boxes"""

        filename = filename or self.filename
        if not filename:
            raise ValueError("`filename` should be specified either on image"
                             " creation or when calling this function.")

        if self._bboxes_normalization_scale == (1, 1):
            normalized = self.normalize_bboxes()
        else:
            normalized = self

        labels, offsets = normalized.labels_and_offsets(default_boxes,
                                                        threshold,
                                                        class_mapping)

        reverse_mapping = reverse_dict(class_mapping)
        classnames = [reverse_mapping[label] for label in labels]

        height, width = self.size
        scale = (1 / height, 1 / width)

        recovered = default_boxes.rescale(scale)
        matched = BoundBoxArray.from_boxes(recovered.as_matrix(), classnames)
        matched = matched[matched.index != "background"]

        plot_with_bboxes(normalized.image, matched,
                         colormap, save_path, filename)
