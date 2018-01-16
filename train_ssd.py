import argparse

import matplotlib
matplotlib.use("pdf")

from networks.ssd.ssd import SSD
from networks.ssd.ssd import TRAIN
from networks.ssd.ssd import OutConvoLayer
from loaders.voc_loader import VOCLoader
from datasets.voc_dataset import VOCDataset
from ops.default_boxes import get_default_boxes


parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--train_images", type=str, required=True,
                    help="path to train images")
parser.add_argument("--train_annotations", type=str, required=True,
                    help="path to train annotations")
parser.add_argument("--test_images", type=str, required=True,
                    help="path to test images")
parser.add_argument("--test_annotations", type=str, required=True,
                    help="path to test annotations")
parser.add_argument("--max_images", type=int,
                    help="maximum number of images to load")
parser.add_argument("--scope", type=str, required=True,
                    help="model scope")

args = parser.parse_args()


class Config:

    scope = args.scope
    model_path = "saved_models/{scope}".format(scope=scope)
    summary_path = "summaries/{scope}".format(scope=scope)

    input_shape = (300, 300, 3)
    weight_decay = 0.0005
    batch_norm_decay = 0.995
    batch_size = 32

    out_layers = [OutConvoLayer(name="out_convo4_3",
                                parent="vgg_16.conv4_3",
                                kernel_size=(3, 3),
                                box_ratios=(1, 1/2, 2, 1/3, 3)),
                  OutConvoLayer(name="out_convo7",
                                parent="conv7",
                                kernel_size=(3, 3),
                                box_ratios=(1, 1/2, 2, 1/3, 3)),
                  OutConvoLayer(name="out_convo8_2",
                                parent="conv8_2",
                                kernel_size=(3, 3),
                                box_ratios=(1, 1/2, 2, 1/3, 3)),
                  OutConvoLayer(name="out_convo9_2",
                                parent="conv9_2",
                                kernel_size=(3, 3),
                                box_ratios=(1, 1/2, 2, 1/3, 3)),
                  OutConvoLayer(name="out_convo10_2",
                                parent="conv10_2",
                                kernel_size=(3, 3),
                                box_ratios=(1, 1/2, 2, 1/3, 3)),
                  OutConvoLayer(name="out_convo11_2",
                                parent="conv11_2",
                                kernel_size=(3, 3),
                                box_ratios=(1, 1/2, 2, 1/3, 3))]

    @staticmethod
    def train_transform(image):
        return (image
                .normalize(255)
                .normalize_bboxes(),
                .random_flip())

    @staticmethod
    def test_transform(image):
        return (image
                .normalize(255)
                .normalize_bboxes())

    classnames = VOCDataset.classnames
    matching_threshold = 0.45
    nms_threshold = 0.45
    neg_pos_ratio = 3
    # on prediction
    max_boxes = 20

    @staticmethod
    def learning_rate_schedule(iteration):

        if iteration < 50000:
            return 1e-4
        elif iteration < 80000:
            return 1e-5
        else:
            return 1e-6

    iterations = 100000
    tune_base = True

    save_interval = 1000
    log_interval = 50
    test_interval = 50

# create config and model
config = Config()
model = SSD(config, resume=False, mode=TRAIN)
# generate default boxes
default_boxes = get_default_boxes(model.out_shapes, model.box_ratios)
# create data loader
loader = VOCLoader(args.train_images,
                   args.train_annotations,
                   args.test_images,
                   args.test_annotations,
                   config.train_transform,
                   config.test_transform,
                   default_boxes=default_boxes,
                   matching_threshold=config.matching_threshold,
                   resize_to=config.input_shape,
                   max_samples=args.max_images)
# fit the model to data
model.fit(loader)
