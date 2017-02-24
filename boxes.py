import os
from collections import namedtuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

BoundBox = namedtuple('BoundBox', ['x_min', 'y_min', 'x_max', 'y_max'])
CenterBox = namedtuple('CenterBox', ['center_x', 'center_y', 'width', 'height'])

def from_boundbox_to_centerbox(box):
    if isinstance(box, CenterBox):
        return box
    elif isinstance(box, BoundBox):
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        center_x = x_min + width/2
        center_y = y_min + height/2
        return CenterBox(
                         center_x=center_x,
                         center_y=center_y,
                         width=width,
                         height=height)
    else:
        raise TypeError('`box` should be either CenterBox or BoundBox.')

def from_centerbox_to_boundbox(box):
    if isinstance(box, BoundBox):
        return box
    elif isinstance(box, CenterBox):
        center_x, center_y, width, height = box
        x_min = center_x - width/2
        x_max = center_x + width/2
        y_min = center_y - height/2
        y_max = center_y + height/2
        return BoundBox(
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_max,
                        y_max=y_max)
    else:
        raise TypeError('`box` should be either CenterBox or BoundBox.')

def plot_boxes(boxes, save_path, name):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    boxes = [from_centerbox_to_boundbox(box) for box in boxes]
    colormap = iter(matplotlib.cm.jet(np.linspace(0, 1, len(boxes))))
    for box in boxes:
        xmin, ymin, xmax, ymax = box 
        bx = (xmin, xmax, xmax, xmin, xmin)
        by = (ymin, ymin, ymax, ymax, ymin)
        ax.plot(bx, by, c=next(colormap), lw=2, alpha=0.5)
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, name))
    plt.close()

def box_scale(k, m):
    s_min = 0.2
    s_max = 0.95
    #equation 4 from paper
    s_k = s_min + (s_max - s_min) * (k - 1.0) / (m - 1.0) 
    return s_k

def feature_map_shape(out_shape):
    return out_shape[1], out_shape[2]

def default_box(i, j, scale, box_ratio, width, height):

    default_w = scale*np.sqrt(box_ratio)
    default_h = scale/np.sqrt(box_ratio)
    center_x = (i + 0.5)/width
    center_y = (j + 0.5)/height

    return CenterBox(
                     center_x=center_x,
                     center_y=center_y,
                     width=default_w,
                     height=default_h)

def get_default_boxes(out_shapes, box_ratios):
    boxes = []

    for out_n, out_shape in enumerate(out_shapes):
        layer_boxes = []
        scale = box_scale(out_n + 1, len(box_ratios))
        out_width, out_height = feature_map_shape(out_shape)
        layer_boxes = [default_box(i, j, scale, box_ratio, out_width, out_height)
                        for box_ratio in box_ratios
                        for j in range(out_height)
                        for i in range(out_width)]
        boxes.append(layer_boxes)
    return boxes

def intersection(box1, box2):
    box1 = from_centerbox_to_boundbox(box1)
    box2 = from_centerbox_to_boundbox(box2)

    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    w1 = x_max1 - x_min1
    h1 = y_max1 - y_min1
    w2 = x_max2 - x_min2
    h2 = y_max2 - y_min2

    intersection_w = (w1 + w2) - (max(x_max1, x_max2) - min(x_min1, x_min2))
    intersection_h = (h1 + h2) - (max(y_max1, y_max2) - min(y_min1, y_min2))

    return max(intersection_w, 0)*max(intersection_h, 0)

def jaccard_overlap(box1, box2):
    intersection_ = intersection(box1, box2)
    box1 = from_boundbox_to_centerbox(box1)
    box2 = from_boundbox_to_centerbox(box2)
    union = box1.width*box1.height + box2.width*box2.height - intersection_
    return intersection_/union if union > 0 else 0 

if __name__ == '__main__':
    
    out_shapes = [
                  (1, 38, 38, 75),
                  (1, 19, 19, 75),
                  (1, 10, 10, 75),
                  (1, 5, 5, 75),
                  (1, 3, 3, 75), 
                  (1, 1, 1, 75)]

    box_ratios = [1, 1/3, 3]
       
    box_set = get_default_boxes(out_shapes, box_ratios)
    for boxes, shape in zip(box_set, out_shapes):
        print('Plotting boxes for shape {shape}. Number of boxes: {n}.'.format(
                shape=shape, n=len(boxes)))
        plot_boxes(boxes, 'default_boxes', ' '.join(str(s) for s in shape))