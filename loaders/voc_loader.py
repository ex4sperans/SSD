import random

import numpy as np

from datasets.voc_dataset import VOCDataset


class VOCLoader:

    def __init__(self, train_images_path, train_annotations_path,
                 test_images_path, test_annotations_path,
                 default_boxes, resize_to, matching_threshold,
                 max_samples=None):

        """Create loader for VOC-like dataset.

        Args:
            path_to_images: path to folder with train .jpg images
            path_to_annotations: path to folder with train annotations
                in .xml format. Annotations assumed to have
                the same names as correspoding images.
            path_to_images: path to folder with test .jpg images
            path_to_annotations: path to folder with test annotations in .xml
                format. Annotations assumed to have the same names as
                correspoding images.
            default_boxes: an BoundBoxArray instance with default boxes.
            resize_to: image size to resize to.
            matching_treshold: IOU threshold to match bboxes to default boxes
            max_samples: maximum number of (image, annotation) pairs to load.
                Suitable for debugging/testing.
        """


        self._train = VOCDataset(train_images_path, train_annotations_path,
                                 max_samples=max_samples, name="VOC_train",
                                 resize_to=resize_to)
        self._test = VOCDataset(test_images_path, test_annotations_path,
                                max_samples=max_samples, name="VOC_test",
                                resize_to=resize_to)

        self.default_boxes = default_boxes
        self.resize_to = resize_to
        self.matching_threshold = matching_threshold

    def process_image(self, image):
        """Construct one element minibatch

        Returns the following tuple:
            images: (height, width, 3)
            labels: (n_default_boxes)
            offsets: (n_default_boxes, 4)
        """

        normalized = (image
                      .normalize(255)
                      .normalize_bboxes())

        labels, offsets = normalized.labels_and_offsets(
                                    default_boxes=self.default_boxes,
                                    threshold=self.matching_threshold,
                                    class_mapping=VOCDataset.class_mapping)

        return (np.array(normalized.image, dtype=np.float32),
                np.array(labels, dtype=np.int32),
                np.array(offsets, dtype=np.float32))

    def train_batch(self, batch_size):
        """Construct new train minibatch

        Returns the following tuple:
            images: (batch_size, height, width, 3)
            labels: (batch_size, n_default_boxes)
            offsets: (batch_size, n_default_boxes, 4)
        """

        images, labels, offsets = zip(*[self.process_image(annotated_image)
                                        for annotated_image in
                                        random.sample(self._train.images,
                                                      batch_size)])

        return (np.stack(images),
                np.stack(labels),
                np.stack(offsets))

    def test_batch(self, batch_size):
        """Construct new test minibatch

        Returns the following tuple:
            images: (batch_size, height, width, 3)
            labels: (batch_size, n_default_boxes)
            offsets: (batch_size, n_default_boxes, 4)
        """

        images, labels, offsets = zip(*[self.process_image(annotated_image)
                                        for annotated_image in
                                        random.sample(self._test.images,
                                                      batch_size)])

        return (np.stack(images),
                np.stack(labels),
                np.stack(offsets))

    def random_train_iterator(self, batch_size, iterations):
        """A generator to produce batches from train set"""

        for iteration in range(iterations):
            yield self.train_batch(batch_size)

    def single_train_image(self):

        image = random.choice(self._train)
        filename = image.filename

        # labels and offsets are omitted 
        normalized, _, _ = self.process_image(image)
        return normalized, image.filename

    def single_test_image(self):

        image = random.choice(self._test)
        filename = image.filename

        # labels and offsets are omitted 
        normalized, _, _ = self.process_image(image)
        return normalized, image.filename
