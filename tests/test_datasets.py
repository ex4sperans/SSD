import os

from datasets.voc_dataset import VOCDataset


DIRNAME = os.path.dirname(__file__)


def test_voc_dataset():

    IMAGES = os.path.join(DIRNAME, "mini_voc/test/images")
    ANNOTATIONS = os.path.join(DIRNAME, "mini_voc/test/annotations")

    dataset = VOCDataset(IMAGES, ANNOTATIONS,
                         resize_to=(300, 300),
                         max_samples=3)

    assert len(dataset) == 3

    for image in dataset:
        assert image.size == (300, 300)
