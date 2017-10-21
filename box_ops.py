import numpy as np

def jaccard_overlap(boxes, target_box):

    boundboxes = boxes.as_boundbox_array()
    target_bbox = target_box.as_boundbox_array()
    centerboxes = boxes.as_centerbox_array()
    target_cbox = target_box.as_centerbox_array()

    w1 = centerboxes.width
    w2 = target_cbox.width 
    h1 = centerboxes.height
    h2 = target_cbox.height

    intersection_w = (w1 + w2) - (np.maximum(boundboxes.x_max, target_bbox.x_max)\
                               - np.minimum(boundboxes.x_min, target_bbox.x_min))
    intersection_h = (h1 + h2) - (np.maximum(boundboxes.y_max, target_bbox.y_max)\
                               - np.minimum(boundboxes.y_min, target_bbox.y_min))

    intersection =  np.maximum(intersection_w, 0)*np.maximum(intersection_h, 0)
    union = w1 * h1 + w2 * h2 - intersection

    # to avoid division by zero
    positive_mask = union > 0
    overlap = (intersection * positive_mask) / union
    return overlap 

