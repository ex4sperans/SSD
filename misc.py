import itertools
import json
import os
import fnmatch

from scipy import ndimage

def flatten_list(nested_list):
    #flatten nested list of arbitrary depth
    if isinstance(nested_list[0], list):
        return flatten_list(list(itertools.chain.from_iterable(nested_list)))
    else:
        return nested_list

def height_and_width(shape):
    if len(shape) == 4:
        return shape[1], shape[2]
    elif len(shape) == 3:
        return shape[0], shape[1]
    else:
        raise ValueError('Could not infer height and'\
            ' width from shape {shape}.'.format(shape=shape))

def load_image(image_path):
        return ndimage.imread(image_path)

def load_json(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

def resize(image, new_shape):
        return misc.imresize(image, new_shape) 

def find_files(path, file_type, sort=True):
    files = list()
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, file_type):
            files.append(os.path.join(root, filename))
    return sorted(files) if sort else files
