import os
import xml.etree.ElementTree as xml_parser
import random
import functools

import numpy as np

import boxes
import misc
import augmentation_ops

class VOCLoader:

    def __init__(self, dataset_params_path='voc_dataset_params.json',
        preprocessing=None, normalization=None, augmentation=None):

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
        self._set_normalization_fn(normalization)
        self._set_augmentation_fn(augmentation)

        print('\nCreated loader for VOC dataset.')
        print('Loaded params from {path}'.format(path=dataset_params_path))
        print('Size of train set: {size}'.format(size=len(self.train_set)))
        print('Size of test set: {size}'.format(size=len(self.test_set)))
        print('Preprocessing: {}'.format(preprocessing))
        print('Normalization: {}'.format(normalization))
        print('Augmentation: {}'.format(augmentation))
        
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

    def new_batch(self, batch_size, dataset, augment=False):
    
        batch = random.sample(dataset, batch_size)
        batch = [(misc.load_image(image), VOCLoader.parse_annotation(annotation))
                for image, annotation in batch]

        if hasattr(self, '_preprocess'):
            batch = [self._preprocess(image, annotation) for image, annotation in batch]

        if hasattr(self, '_normalize'):
            batch = [(self._normalize(image), annotation) for image, annotation in batch]

        if hasattr(self, '_augment') and augment:
            batch = [self._augment(image, annotation) for image, annotation in batch]            

        return batch

    def new_train_batch(self, batch_size, augment=True):
        return self.new_batch(batch_size, self.train_set, augment=augment)

    def new_test_batch(self, batch_size):
        return self.new_batch(batch_size, self.test_set)

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
        elif preprocessing is None:
            pass
        else:
            raise TypeError('`preprocessing` have to be an instance of tuple.')

    def _set_normalization_fn(self, normalization):
        if isinstance(normalization, str):
            if normalization == 'divide_255':

                def divide_255(image):
                    return image/255

                self._normalize = divide_255

        elif normalization is None:
            pass
        else:
            raise TypeError('`normalization` have to be an instance of string.')

    def _set_augmentation_fn(self, augmentation):

        augmentations = {'random_hflip': augmentation_ops.random_hflip,
                         'random_vflip': augmentation_ops.random_vflip,
                         'random_tile': augmentation_ops.random_tile,
                         'random_crop': augmentation_ops.random_crop}

        if isinstance(augmentation, list):

            def augment(image, annotation):
                            
                for augmentation_type, probability in augmentation:
                    if np.random.uniform() < probability:
                        image, annotation['objects'] = \
                            augmentations[augmentation_type](image, annotation['objects'])

                annotation['file_name'] = 'augmented_' + annotation['file_name']

                return image, annotation

            self._augment = augment
        elif augmentation is None:
            pass
        else:
            raise TypeError('`augmentation` have to be an instance of list.')
