from math import sqrt

from ops.misc import height_and_width, flatten_list
from containers.box_arrays import BoundBoxArray


def box_scale(k, m, s_min=0.2, s_max=0.9):
    """Computes box scale as function of k
    
    Args:
        k: current layer number
        m: total number layers of
    
    Returns:
        float, box scale"""

    # equation 4 from paper
    return s_min + (s_max - s_min) * (k - 1) / (m - 1)


def get_default_box(i, j, scale, box_ratio, width, height):
    """Create default centerbox for given position (i, j)
    
    Args:
        i: position on y axis
        j: position on x axis
        scale: box scale
        box_ratio: box aspect ratio
        width: width of current feature map
        height: height of current feature map
    
    Returns:
        a tuple with (center_x, center_y, default_w, default_h)
            of the default box
    """

    default_w = scale * sqrt(box_ratio)
    default_h = scale / sqrt(box_ratio)
    center_x = (j + 0.5) / width
    center_y = (i + 0.5) / height

    return (center_x,
            center_y,
            default_w,
            default_h)


def get_default_boxes(out_shapes, box_ratios):
    """Returns BoundBoxArray of default boxes
    
    Args:
        out_shapes: a list of tuples with output shapes
        box_ration: a list of box aspect ratios
    
    Returns:
    """

    default_boxes = []
    n_outs = len(out_shapes)
    scales = (box_scale(n_out + 1, n_outs) for n_out in range(n_outs))

    layer_params = zip(out_shapes, scales, box_ratios)
    for out_shape, scale, layer_box_ratios in layer_params:
        height, width = height_and_width(out_shape)
        layer_boxes = [[[get_default_box(i, j, scale, box_ratio, width, height)
                         for box_ratio in layer_box_ratios]
                        for i in range(height)]
                       for j in range(width)]
        default_boxes.append(layer_boxes)

    return BoundBoxArray.from_centerboxes(flatten_list(default_boxes)).clip()
