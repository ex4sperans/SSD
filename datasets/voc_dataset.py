from collections import OrderedDict

from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm


from ops.io_ops import find_files
from containers.image import AnnotatedImage


class VOCDataset:

    class_mapping = OrderedDict([("pottedplant", 1),
                                 ("boat", 2),
                                 ("aeroplane", 3),
                                 ("motorbike", 4),
                                 ("cow", 5),
                                 ("sheep", 6),
                                 ("bus", 7),
                                 ("bicycle", 8),
                                 ("person", 9),
                                 ("car", 10),
                                 ("diningtable", 11),
                                 ("cat", 12),
                                 ("tvmonitor", 13),
                                 ("sofa", 14),
                                 ("bird", 15),
                                 ("train", 16),
                                 ("bottle", 17),
                                 ("dog", 18),
                                 ("chair", 19),
                                 ("horse", 20)]
                               )
    # background class
    background = 0

    classnames = list(class_mapping.keys())
    number_of_classes = len(classnames)

    colormap = dict(zip(classnames, np.linspace(0, 1, number_of_classes)))

    def __init__(self, path_to_images, path_to_annotations,
                 resize_to=None, name="VOC", max_samples=None):
        """Create VOC-like dataset.

        Args:
            path_to_images: path to folder with .jpg images
            path_to_annotations: path to folder with annotations in .xml
                format. Annotations assumed to have the same names as
                correspoding images.
            resize_to: image size to resize to.
            name: optional name for dataset
            max_samples: maximum number of (image, annotation) pairs to load.
                Suitable for debugging/testing.
        """

        self.name = name

        images = sorted(find_files(path_to_images, "*.jpg"))
        annotations = sorted(find_files(path_to_annotations, "*.xml"))

        if max_samples:
            images = images[:max_samples]
            annotations = annotations[:max_samples]

        self.images = []

        for image, annotation in tqdm(zip(images, annotations), ncols=75,
                                      desc="Loading {}".format(self.name),
                                      total=len(images)):
            image = AnnotatedImage.load(image, annotation)

            if resize_to:
                image = image.resize(resize_to)

            self.images.append(image)

    def __iter__(self):
        return iter(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return self.images[item]