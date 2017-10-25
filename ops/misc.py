import itertools
import json
import os
import fnmatch

from scipy.ndimage import imread
from skimage.transform import resize as imresize
from skimage.transform import rotate as imrotate
from skimage.io import imsave


def reverse_dict(d):
    return dict((v, k) for k, v in d.items())


def flatten_list(nested_list):
    #flatten nested list of arbitrary depth
    if isinstance(nested_list[0], list):
        return flatten_list(list(itertools.chain.from_iterable(nested_list)))
    else:
        return nested_list


def height_and_width(shape):
    """Retrieve height and width from given shape"""
    if len(shape) == 4:
        return shape[1], shape[2]
    elif len(shape) == 3:
        return shape[0], shape[1]
    elif len(shape) == 2:
        return tuple(shape)
    else:
        raise ValueError("Could not infer height and "
                         "width from shape {shape}."
                         .format(shape=shape))


def load_image(image_path):
    return imread(image_path)


def resize(image, new_shape):
        return imresize(image, new_shape, preserve_range=True) 


def rotate(image, angle):
    return imrotate(image, angle, preserve_range=True)


def find_files(path, file_type, sort=True):
    files = list()
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, file_type):
            files.append(os.path.join(root, filename))
    return sorted(files) if sort else files


def file_name_with_extension(file_path):
    head, tail = os.path.split(file_path)
    return tail


def file_name(file_path):
    file_name_with_ext = file_name_with_extension(file_path)
    root, ext = os.path.splitext(file_name_with_ext)
    return root
