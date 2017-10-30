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

args = parser.parse_args()


class Config:

    model_path = "saved_models/SSD"
    summary_path = "summaries"

    input_shape = (300, 300, 3)
    weight_decay = 0.0005
    batch_norm_decay = 0.995
    momentum = 0.9
    batch_size = 32

    out_layers = [OutConvoLayer(name="out_convo4_3",
                                parent="vgg_16.conv4_3",
                                kernel_size=(3, 3),
                                box_ratios=(1,)),
                  OutConvoLayer(name="out_convo7",
                                parent="conv7",
                                kernel_size=(3, 3),
                                box_ratios=(1,)),
                  OutConvoLayer(name="out_convo9_2",
                                parent="conv9_2",
                                kernel_size=(3, 3),
                                box_ratios=(1,))]

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

    save_interval = 1000
    log_interval = 1
    test_interval = 50

# create config and model
config = Config()
model = SSD(config, resume=False, mode=TRAIN)
# generate default boxes
default_boxes = get_default_boxes(model.out_shapes, model.box_ratios)

loader = VOCLoader(args.train_images,
                   args.train_annotations,
                   args.test_images,
                   args.test_annotations,
                   default_boxes=default_boxes,
                   matching_threshold=config.matching_threshold,
                   resize_to=config.input_shape)
# fit the model to data
model.fit(loader)