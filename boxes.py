import os
from collections import namedtuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import misc

"""There are two box classes: BoundBox and CenterBox. BoundBox is defined by
the coordinates of the top left corner and the bottom right corner. CenterBox is 
defined by the box center, box width and box height. Model is trained in CenterBox 
coordinates (more precisely, with offsets, but still in CenterBox coordinates), 
while the vast majority of post- and pre- processing is performed using BoundBox.
Two functions defined below perform conversion between those two boxes types.
Both box types are namedtuple`s, therefore they have explicit and clear names for
their fields.
"""

BoundBox = namedtuple('BoundBox', ['x_min', 'y_min', 'x_max', 'y_max'])
CenterBox = namedtuple('CenterBox', ['center_x', 'center_y', 'width', 'height'])

def boundbox_to_centerbox(box):
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
        raise TypeError('`box` should be either CenterBox or BoundBox.'\
            ' But `box` is of type {box_type}.'.format(box_type=type(box)))

def centerbox_to_boundbox(box):
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
        raise TypeError('`box` should be either CenterBox or BoundBox.'\
            ' But `box` is of type {box_type}.'.format(box_type=type(box)))

def plot_default_boxes(boxes, save_path, name):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    boxes = [centerbox_to_boundbox(box) for box in boxes]
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
    s_min = 0.1
    s_max = 0.9
    #equation 4 from paper
    s_k = s_min + (s_max - s_min) * (k - 1.0) / (m - 1.0) 
    return s_k

def height_and_width(shape):
    if len(shape) == 4:
        return shape[1], shape[2]
    elif len(shape) == 3:
        return shape[0], shape[1]
    else:
        raise ValueError('Could not infer height and'\
            ' width from shape {shape}.'.format(shape=shape))

def clip_box(box, maxval=1, minval=0):
    box = centerbox_to_boundbox(box)
    box = BoundBox(
                   x_min=max(box.x_min, minval),
                   y_min=max(box.y_min, minval),
                   x_max=min(box.x_max, maxval),
                   y_max=min(box.y_max, maxval))
    return boundbox_to_centerbox(box)

def default_box(i, j, scale, box_ratio, width, height):

    default_w = scale*np.sqrt(box_ratio)
    default_h = scale/np.sqrt(box_ratio)
    center_x = (i + 0.5)/width
    center_y = (j + 0.5)/height

    return clip_box(CenterBox(
                     center_x=center_x,
                     center_y=center_y,
                     width=default_w,
                     height=default_h))

def get_default_boxes(out_shapes, box_ratios):
    default_boxes = []
    n_outs = len(out_shapes)
    scales = [box_scale(n_out + 1, n_outs) for n_out in range(n_outs)]
    layer_params = zip(out_shapes, scales, box_ratios)
    for out_shape, scale, layer_box_ratios in layer_params:
        out_height, out_width = misc.height_and_width(out_shape)
        layer_boxes = [[[default_box(i, j, scale, box_ratio, out_width, out_height)
                        for box_ratio in layer_box_ratios]
                        for j in range(out_height)]
                        for i in range(out_width)]
        default_boxes.append(layer_boxes)
    print('\nNumber of default boxes: {n}.'.format(n=len(misc.flatten_list(default_boxes))))
    return default_boxes

def intersection(box1, box2):
    box1 = centerbox_to_boundbox(box1)
    box2 = centerbox_to_boundbox(box2)

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
    box1 = boundbox_to_centerbox(box1)
    box2 = boundbox_to_centerbox(box2)
    union = box1.width*box1.height + box2.width*box2.height - intersection_
    return intersection_/union if union > 0 else 0 

def normalize_box(box, height, width):
    #normalize width and height of a box to be in range (0, 1)
    box = centerbox_to_boundbox(box)
    box = BoundBox(
                   x_min=box.x_min/width,
                   y_min=box.y_min/height,
                   x_max=box.x_max/width,
                   y_max=box.y_max/height)
    return box

def recover_box(box, height, width):
    #recovers a box (translates ratios to pixels)
    box = centerbox_to_boundbox(box)
    box = BoundBox(
                   x_min=int(box.x_min*width),
                   y_min=int(box.y_min*height),
                   x_max=int(box.x_max*width),
                   y_max=int(box.y_max*height))
    return box

def resize_box(box, orig_height, orig_width, new_height, new_width):
    normalized_box = normalize_box(box, orig_height, orig_width)
    return recover_box(normalized_box, new_height, new_width) 


def plot_with_bboxes(image, save_path, file_name, 
                    bboxes, ground_truth_boxes):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image)

    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox 
        bx = (xmin, xmax, xmax, xmin, xmin)
        by = (ymin, ymin, ymax, ymax, ymin)
        ax.plot(bx, by, c='b', lw=1)

    for gtbox in ground_truth_boxes:
        xmin, ymin, xmax, ymax = gtbox 
        bx = (xmin, xmax, xmax, xmin, xmin)
        by = (ymin, ymin, ymax, ymax, ymin)
        ax.plot(bx, by, c='g', lw=1.5)
        
    ax.set_axis_off()    
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, file_name))
    plt.close()

def plot_predicted_bboxes(image, save_path, file_name, 
                          bboxes, labels, confidences, class_names):

    if len(bboxes) != len(labels):
        raise ValueError('Each label should correspond to bbox.')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image)

    colors = list(matplotlib.cm.hsv(np.linspace(0, 1, len(class_names))))

    for box, label, confidence in zip(bboxes, labels, confidences):
        box = centerbox_to_boundbox(box)
        xmin, ymin, xmax, ymax = box 
        bx = (xmin, xmax, xmax, xmin, xmin)
        by = (ymin, ymin, ymax, ymax, ymin)
        color = colors[class_names.index(label)]
        ax.plot(bx, by, c=color, lw=1.5)
        bbox_props = dict(boxstyle='square,pad=0.3',
                         fc=color, ec=color, lw=1)
        text = '{} {:.2f}'.format(label, confidence)
        ax.text(xmin, ymin, text, ha='center', va='center',
                 size=8, color='black', bbox=bbox_props)
    
    ax.set_axis_off()    
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, file_name))
    plt.close()

    
if __name__ == '__main__':
    
    out_shapes = [
                  (1, 38, 38, 75),
                  (1, 19, 19, 75),
                  (1, 10, 10, 75),
                  (1, 5, 5, 75),
                  (1, 3, 3, 75), 
                  (1, 1, 1, 75)]

    box_ratios = [[1, 1/2, 2]]*len(out_shapes)
       
    box_set = get_default_boxes(out_shapes, box_ratios)

    for boxes, shape in zip(box_set, out_shapes):
        boxes = misc.flatten_list(boxes)
        print('Plotting boxes for shape {shape}. Number of boxes: {n}.'.format(
                shape=shape, n=len(boxes)))
        plot_default_boxes(boxes, 'default_boxes', ' '.join(str(s) for s in shape))