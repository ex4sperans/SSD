import os
import xml.etree.ElementTree as xml_parser
import random
import functools

import numpy as np

import boxes
import misc

class VOCLoader:

    def __init__(self, dataset_params_path='voc_dataset_params.json', preprocessing=None):

        self.dataset_params = misc.load_json(dataset_params_path)
        self.train_images = misc.find_files(
            self.dataset_params['train']['images'], '*.jpg')
        self.train_annotations = misc.find_files(
            self.dataset_params['train']['annotations'], '*.xml')
        self.test_images = misc.find_files(
            self.dataset_params['test']['images'], '*.jpg')
        self.test_annotations = misc.find_files(
            self.dataset_params['test']['annotations'], '*.xml')

        self.train_set = list(zip(self.train_images, self.train_annotations))
        self.test_set = list(zip(self.test_images, self.test_annotations))

        self._set_preprocessing_fn(preprocessing)
        
    @staticmethod
    def parse_annotation(annotation_path):
        root = xml_parser.parse(annotation_path).getroot()
        annotation = dict()
        annotation['file_name'] = root.find('filename').text
        annotation['objects'] = list()
        for obj in root.findall('object'):
            #xmin, ymin, xmax, ymax
            bbox = boxes.BoundBox(*[int(coord.text) for coord in obj.find('bndbox')])
            annotation['objects'].append((obj.find('name').text, bbox))
        return annotation

    def new_batch(self, batch_size):

        batch = random.sample(self.train_set, batch_size)
        batch = [(misc.load_image(image), VOCLoader.parse_annotation(annotation))
                for image, annotation in batch]

        if hasattr(self, '_preprocess'):
            batch = [self._preprocess(image, annotation) for image, annotation in batch]

        return batch

    def _set_preprocessing_fn(self, preprocessing):
        if isinstance(preprocessing, tuple):
            preprocessing_type, preprocessing_params = preprocessing
            if preprocessing_type == 'resize':
                height, width = misc.height_and_width(preprocessing_params)
                def _preprocess(image, annotation):
                    image_height, image_width = misc.height_and_width(image.shape)
                    image = misc.resize(image, preprocessing_params)
                    process_box = functools.partial(
                                                    boxes.resize_box,
                                                    orig_height=image_height,
                                                    orig_width=image_width,
                                                    new_height=height,
                                                    new_width=width)
                    annotation['objects'] = [(class_name, process_box(box))
                                            for class_name, box in annotation['objects']]
                    return image, annotation
                self._preprocess = _preprocess