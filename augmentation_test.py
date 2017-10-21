import random

import numpy as np

import voc_loader
import boxes
import misc

loader = voc_loader.VOCLoader(
                              preprocessing=('resize', (300, 300, 3)),
                              normalization='divide_255')

image, annotation = loader.new_train_batch(1)[0]

def random_crop(image, annotation):

    height, width = misc.height_and_width(image.shape)

    x_min = int(random.uniform(0, 0.3)*width)
    y_min = int(random.uniform(0, 0.3)*height)
    x_max = int(random.uniform(0.7, 1)*width)
    y_max = int(random.uniform(0.7, 1)*height)

    patch = boxes.BoundBox(x_min, y_min, x_max, y_max)

    valid_annotations = [(name, bbox) for name, bbox in annotation if
            x_max > boxes.boundbox_to_centerbox(bbox).center_x > x_min and
            y_max > boxes.boundbox_to_centerbox(bbox).center_y > y_min]

    cropped_image = image[y_min:y_max, x_min:x_max]
    new_height, new_width = misc.height_and_width(cropped_image.shape)
    new_image = misc.resize(cropped_image, image.shape)

    if not valid_annotations:
        return new_image, []

    def fit_box(bbox):
        bbox = boxes.clip_box(bbox, x_min, y_min, x_max, y_max)
        bbox = boxes.centerbox_to_boundbox(bbox)
        bbox = boxes.shift_box(bbox, x_min, y_min)
        bbox = boxes.resize_box(bbox, new_height, new_width, height, width)
        return bbox        

    new_annotation = [(name, fit_box(bbox))
                      for name, bbox in valid_annotations]
    
    return new_image, new_annotation

def random_flip(image, annotation, probability):

    if random.uniform(0, 1) > probability:
        return image, annotation
    else:
        flipped_image = np.fliplr(image)
        height, width = misc.height_and_width(image.shape)
        flipped_annotation = [(name, boxes.hflip_box(bbox, height, width)) 
                                          for name, bbox in annotation]
        return flipped_image, flipped_annotation

def random_tile(image, annotation, min_tiles=2, max_tiles=4):

    #sample the number of tiles
    tiles = random.randrange(min_tiles, max_tiles + 1)
    height, width = misc.height_and_width(image.shape)
    initial_shape = image.shape
    tiled_image = np.tile(image, (tiles, tiles, 1))
    tiled_annotation = [(name, boxes.resize_box(
                        boxes.shift_box(bbox, -i*width, -j*height),
                        height*tiles, width*tiles, height, width))
                        for i in range(tiles)
                        for j in range(tiles) 
                        for name, bbox in annotation]
    return misc.resize(tiled_image, initial_shape), tiled_annotation

boxes.plot_with_bboxes(
                       image,
                       'augmentation_plots',
                       annotation['file_name'],
                       [],
                       [bbox for obj, bbox in annotation['objects']])

image, annotation['objects'] = random_crop(image, annotation['objects'])

boxes.plot_with_bboxes(
                       image,
                       'augmentation_plots',
                       'cropped_' + annotation['file_name'],
                       [],
                       [bbox for obj, bbox in annotation['objects']])

image, annotation['objects'] = random_flip(image, annotation['objects'], 1)

boxes.plot_with_bboxes(
                       image,
                       'augmentation_plots',
                       'flipped_' + annotation['file_name'],
                       [],
                       [bbox for obj, bbox in annotation['objects']])

image, annotation['objects'] = random_tile(image, annotation['objects'], 1)

boxes.plot_with_bboxes(
                       image,
                       'augmentation_plots',
                       'tiled_' + annotation['file_name'],
                       [],
                       [bbox for obj, bbox in annotation['objects']])

