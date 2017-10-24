import os
import fnmatch

from scipy.misc import imread
import xml.etree.ElementTree as xml_parser

from containers.box_arrays import BoundBoxArray


def parse_annotation(annotation):
    """Parses PascalVOC-like .xml annotation. BoundBoxArray is returned"""

    root = xml_parser.parse(annotation).getroot()

    boxes = list()
    classnames = list()

    for obj in root.findall('object'):
        # xmin, ymin, xmax, ymax
        boxes.append([int(coord.text) for coord in obj.find('bndbox')])
        classnames.append(obj.find('name').text)

    return BoundBoxArray.from_boundboxes(boxes, classnames)


def load_image(image):
    """Loads image"""

    return imread(image)


def find_files(path, pattern):
    """A generator to produce files matching `pattern` in given
    directory and subdirectories"""

    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, pattern):
            yield os.path.join(root, filename)
