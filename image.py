from scipy.misc import imresize

from misc import height_and_width
from io_ops import parse_annotation, load_image
from plotting import plot_image, plot_with_bboxes


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

    def resize(self, size):
        """Resize image and bboxes according to `size`"""
        new_height, new_width = size
        height, width = height_and_width(self.shape)
        scale = (height / new_height, width / new_width)

        return AnnotatedImage(imresize(self._image, size),
                              self._bboxes.rescale(scale),
                              self._filename)

    def plot(self, save_path, filename=None):
        """Plots and save image"""

        filename = filename or self.filename
        if not filename:
            raise ValueError("`filename` should be specified either "
                             "on image creation or when calling this function.")

        plot_image(self.image, save_path, filename)
    
    def plot_bboxes(self, save_path, filename=None):
        """Plots and save image with bounding boxes"""

        filename = filename or self.filename
        if not filename:
            raise ValueError("`filename` should be specified either "
                             "on image creation or when calling this function.")

        plot_with_bboxes(self.image, self.bboxes, save_path, filename)