import numpy as np

import ssd
import boxes
import voc_loader

overlap_threshold = 0.5
batch_size = 8
n_iter = 100
learning_rate = 0.001

model = ssd.SSD()
loader = voc_loader.VOCLoader(preprocessing=('resize', model.input_shape))
default_boxes = boxes.get_default_boxes(model.out_shapes, model.box_ratios)

model.train(loader, default_boxes, overlap_threshold, batch_size, learning_rate, n_iter)