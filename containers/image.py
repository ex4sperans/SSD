import numpy as np
from scipy.misc import imresize

from ops.misc import height_and_width, file_name, reverse_dict
from ops.io_ops import parse_annotation, load_image
from ops.plotting import plot_image, plot_with_bboxes
from ops.box_ops import calculate_offsets
from ops import augmentation
from containers.box_arrays import BoundBoxArray


class AnnotatedImage:

    @classmethod
    def load(cls, image, annotation):
        """Initialize AnnotatedImage from .jpg image and .xml annotation"""

        return cls(image=load_image(image),
                   bboxes=parse_annotation(annotation),
                   filename=file_name(image))

    def __init__(self, image, bboxes, filename=None,
                 bboxes_normalized=False):
        """Initialize AnnotatedImage object"""

        self._image = image
        self._bboxes = bboxes
        self._filename = filename
        # keep scale to be able to convert
        # normalized bboxes to image scale
        self._bboxes_normalized = bboxes_normalized

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

        return AnnotatedImage(np.divide(self._image, scale, dtype=np.float32),
                              self.bboxes,
                              self.filename,
                              self._bboxes_normalized)

    def normalize_bboxes(self):
        """Normalize bboxes to be in range of (0, 1) for both axes"""

        if not self._bboxes_normalized:
            return AnnotatedImage(self.image,
                                  self.bboxes.rescale(self.size),
                                  self.filename,
                                  bboxes_normalized=True)
        else:
            return self

    def recover_bboxes(self):
        """Recover bboxes to represent size in pixels"""

        if self._bboxes_normalized:
            height, width = self.size
            scale = (1 / height, 1 / width)

            return AnnotatedImage(self.image,
                                  self.bboxes.rescale(scale),
                                  self.filename,
                                  bboxes_normalized=False)
        else:
            return self

    def resize(self, size):
        """Resize image and bboxes according to `size`"""
        new_height, new_width = height_and_width(size)
        height, width = height_and_width(self.shape)
        scale = (height / new_height, width / new_width)

        return AnnotatedImage(imresize(self.image, size),
                              self.bboxes.rescale(scale),
                              self.filename,
                              self._bboxes_normalized)

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
                              for classname in self.bboxes.classnames],
                             dtype=np.int32)
        labels = (matched * classmask).sum(axis=1, dtype=np.int32)

        # offsets
        default_matched = matched.any(axis=1)
        box_matched = matched[default_matched].argmax(axis=1)
        compressed_offsets = calculate_offsets(default_boxes[default_matched],
                                               self.bboxes.iloc[box_matched])
        offsets = np.zeros((default_matched.size, 4), dtype=np.float32)
        offsets[default_matched] = compressed_offsets

        return labels, offsets

    def random_hflip(self, probability=0.5):
        """Flips image horizontally with probability `probability`"""

        if np.random.uniform() < probability:
            return augmentation.hflip(self)
        else:
            return self

    def random_crop(self, probability=0.25):
        """Crops images with probability `probability`"""

        if np.random.uniform() < probability:

            x_min = np.random.uniform(0, 0.3)
            x_max = np.random.uniform(0.7, 1.0)
            y_min = np.random.uniform(0, 0.3)
            y_max = np.random.uniform(0.7, 1.0)

            return augmentation.crop(self, (x_min, y_min, x_max, y_max))
        else:
            return self

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
                             class_mapping, colormap, background_class,
                             filename=None):
        """Plot and save image with matching bounding boxes"""

        filename = filename or self.filename
        if not filename:
            raise ValueError("`filename` should be specified either on image"
                             " creation or when calling this function.")

        normalized = self if self._bboxes_normalized else self.normalize_bboxes()

        labels, offsets = normalized.labels_and_offsets(default_boxes,
                                                        threshold,
                                                        class_mapping)

        reverse_mapping = reverse_dict(class_mapping)
        reverse_mapping[background_class] = "background"
        classnames = [reverse_mapping[label] for label in labels]

        # recover default boxes
        height, width = self.size
        scale = (1 / height, 1 / width)
        recovered = default_boxes.rescale(scale)

        matched = BoundBoxArray.from_boxes(recovered.as_matrix(), classnames)        
        matched = matched[matched.classnames != "background"]

        plot_with_bboxes(normalized.image, matched,
                         colormap, save_path, filename)
