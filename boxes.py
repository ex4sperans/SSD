import os
from collections import namedtuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

boundbox = namedtuple('boundbox', ['x_min', 'y_min', 'x_max', 'y_max'])
centerbox = namedtuple('centerbox', ['center_x', 'center_y', 'width', 'height'])

def from_boundbox_to_centerbox(bound_box):
    x_min, y_min, x_max, y_max = bound_box
    width = x_max - x_min
    height = y_max - y_min
    center_x = x_min + width/2
    center_y = y_min + height/2
    return centerbox(
                     center_x=center_x,
                     center_y=center_y,
                     width=width,
                     height=height)

def from_centerbox_to_boundbox(center_box):
    center_x, center_y, width, height = center_box
    x_min = center_x - width/2
    x_max = center_x + width/2
    y_min = center_y - height/2
    y_max = center_y + height/2
    return boundbox(
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max)

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

    return (center_x, center_y, default_w, default_h)

def get_default_boxes(out_shapes, box_ratios):
    boxes = []

    for o_i, out_shape in enumerate(out_shapes):
        layer_boxes = []
        scale = box_scale(o_i + 1, len(box_ratios))
        out_width, out_height = feature_map_shape(out_shape)
        layer_boxes = [default_box(i, j, scale, box_ratio, out_width, out_height)
                        for box_ratio in box_ratios
                        for j in range(out_height)
                        for i in range(out_width)]

        boxes.append(layer_boxes)
    return boxes

if __name__ == '__main__':
    
    out_shapes = [
                  (1, 38, 38, 75),
                  (1, 19, 19, 75),
                  (1, 10, 10, 75),
                  (1, 5, 5, 75),
                  (1, 3, 3, 75), 
                  (1, 1, 1, 75)]

    box_ratios = [1, 1/3, 3]
       
    boxes_set = get_default_boxes(out_shapes, box_ratios)
    for boxes, shape in zip(boxes_set, out_shapes):
        print('Plotting boxes for shape {shape}. Number of boxes: {n}.'.format(
                shape=shape, n=len(boxes)))
        plot_boxes(boxes, 'boxes', ' '.join(str(s) for s in shape))