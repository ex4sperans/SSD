from collections import OrderedDict

from matplotlib import pyplot as plt
import numpy as np


from ops.io_ops import find_files
from containers.image import AnnotatedImage


class VOCDataset:

    class_mapping = OrderedDict(background=0,
                                pottedplant=1,
                                boat=2,
                                aeroplane=3,
                                motorbike=4,
                                cow=5,
                                sheep=6,
                                bus=7,
                                bicycle=8,
                                person=9,
                                car=10,
                                diningtable=11,
                                cat=12,
                                tvmonitor=13,
                                sofa=14,
                                bird=15,
                                train=16,
                                bottle=17,
                                dog=18,
                                chair=19,
                                horse=20
                               )

    classnames = list(class_mapping.keys())
    number_of_classes = len(classnames)

    colormap = dict(zip(classnames, np.linspace(0, 1, number_of_classes)))

    def __init__(self, path_to_images, path_to_annotations):

        images = sorted(find_files(path_to_images, "*.jpg"))[:5]
        annotations = sorted(find_files(path_to_annotations, "*.xml"))[:5]

        self.images = [AnnotatedImage.load(image, annotation)
                       for image, annotation in zip(images, annotations)]
