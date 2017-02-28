import json
import os
import xml.etree.ElementTree as xml_parser
import fnmatch
import random
import functools

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy import ndimage

from boxes import BoundBox, CenterBox

class VOCLoader:

    def __init__(self, dataset_params_path='voc_dataset_params.json', preprocessing=None):

        self.dataset_params = VOCLoader.load_json(dataset_params_path)
        self.train_images = VOCLoader.find_files(
            self.dataset_params['train']['images'], '*.jpg')
        self.train_annotations = VOCLoader.find_files(
            self.dataset_params['train']['annotations'], '*.xml')
        self.test_images = VOCLoader.find_files(
            self.dataset_params['test']['images'], '*.jpg')
        self.test_annotations = VOCLoader.find_files(
            self.dataset_params['test']['annotations'], '*.xml')

        self.train_set = list(zip(self.train_images, self.train_annotations))
        self.test_set = list(zip(self.test_images, self.test_annotations))

        self._set_preprocessing_fn(preprocessing)
        
    @staticmethod  
    def find_files(path, file_type, sort=True):
        files = list()
        for root, dirnames, filenames in os.walk(path):
            for filename in fnmatch.filter(filenames, file_type):
                files.append(os.path.join(root, filename))
        return sorted(files) if sort else files

    @staticmethod
    def parse_annotation(annotation_path):
        root = xml_parser.parse(annotation_path).getroot()
        annotation = dict()
        annotation['file_name'] = root.find('filename').text
        annotation['objects'] = list()
        for obj in root.findall('object'):
            #xmin, ymin, xmax, ymax
            bbox = BoundBox(*[int(coord.text) for coord in obj.find('bndbox')])
            annotation['objects'].append((obj.find('name').text, bbox))
        return annotation

    def new_batch(self, batch_size):

        batch = random.sample(self.train_set, batch_size)
        batch = [(VOCLoader.load_image(image), VOCLoader.parse_annotation(annotation))
                for image, annotation in batch]

        if hasattr(self, '_preprocess'):
            batch = [(self._preprocess(image), annotation)
                    for image, annotation in batch]

        return batch

    def _set_preprocessing_fn(self, preprocessing):
        if isinstance(preprocessing, tuple):
            preprocessing_type, preprocessing_params = preprocessing
            if preprocessing_type == 'resize':
                self._preprocess = functools.partial(
                    VOCLoader.resize, new_shape=preprocessing_params)

    @staticmethod
    def plot_with_bbox(image, annotation, save_path):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(image)

        for class_name, bbox in annotation['objects']:
            xmin, ymin, xmax, ymax = bbox 
            bx = (xmin, xmax, xmax, xmin, xmin)
            by = (ymin, ymin, ymax, ymax, ymin)
            ax.plot(bx, by, c='b', lw=2)
            
        ax.set_axis_off()    
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, annotation['file_name']))
        plt.close()

    @staticmethod
    def load_image(image_path):
        return ndimage.imread(image_path)

    @staticmethod
    def load_json(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data

    @staticmethod
    def resize(image, new_shape):
        return misc.imresize(image, new_shape) 
