from scipy.misc import imresize

from misc import height_and_width


class AnnotatedImage: 

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

