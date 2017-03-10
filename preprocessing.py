import numpy as np

import boxes
import misc
from matching import match_boxes

def offsets(ground_truth_box, default_box):

    #offsets based on differences
    ground_truth_box = boxes.boundbox_to_centerbox(ground_truth_box)
    default_box = boxes.boundbox_to_centerbox(default_box)

    return boxes.CenterBox(
                           ground_truth_box.center_x - default_box.center_x,
                           ground_truth_box.center_y - default_box.center_y,
                           ground_truth_box.width - default_box.width,
                           ground_truth_box.height - default_box.height)

def log_offsets(ground_truth_box, default_box):

    #offsets based on ratios
    ground_truth_box = boxes.boundbox_to_centerbox(ground_truth_box)
    default_box = boxes.boundbox_to_centerbox(default_box)

    return boxes.CenterBox(
                           (ground_truth_box.center_x - default_box.center_x)/default_box.width,
                           (ground_truth_box.center_y - default_box.center_y)/default_box.height,
                           np.log(ground_truth_box.width/default_box.width),
                           np.log(ground_truth_box.height/default_box.height))

def process_matches(matches, default_boxes, class_names):

    background_class = len(class_names)

    def offsets_and_class(default_box):
        for matched_default_box, (ground_truth_box, class_name) in matches:
            if default_box == matched_default_box:
                return (log_offsets(ground_truth_box, default_box),
                        class_names.index(class_name))
        return boxes.CenterBox(0, 0, 0, 0), background_class

    feed = [[[[offsets_and_class(default_box)
               for default_box in y_boxes] 
               for y_boxes in x_boxes]
               for x_boxes in out_boxes]
               for out_boxes in default_boxes]

    return feed

def get_feed(batch, model, default_boxes, threshold):
    images = [image for image, annotation in batch]

    matches_batch = []
    for image, annotation in batch:
        matches = match_boxes(
                              annotations=annotation['objects'],
                              image=image,
                              default_boxes=default_boxes,
                              out_shapes=model.out_shapes,
                              threshold=threshold)
        matches_batch.append(matches)
        
    feed = [misc.flatten_list(process_matches(
                                              matches=matches,
                                              default_boxes=default_boxes,
                                              class_names=model.class_names))
                for matches in matches_batch]

    # split feed batch into offsets and labels batches
    offsets, labels = list(zip(*[list(zip(*f)) for f in feed]))
    return images, list(offsets), list(labels)

def positives_and_negatives(confidences, labels, model, neg_pos_ratio):

    background_class = model.n_classes
    positives = np.not_equal(labels, background_class).astype(np.float32)
    # calculate the number of matched boxes for each element of batch
    n_positives = np.sum(positives, 1)
    n_negatives = n_positives*neg_pos_ratio
    # choose top confidence for each default box
    top_confidences = np.amax(confidences, 2)
    # sort confidences and take the highest among all default boxes
    # boolean mask is applied to skip confidences with positive indicies
    sorted_confidences = np.sort(top_confidences*np.logical_not(positives), 1)
    # reverse the sequence as sort produces indicies in the ascending order
    sorted_confidences = np.flip(sorted_confidences, 1)
    rows = np.arange(len(sorted_confidences), dtype=np.int32)
    columns = n_negatives.astype(np.int32) 
    threshold_confidences = np.expand_dims(sorted_confidences[rows, columns], 1)
    # choose negatives to be all boxes with confidence higher than threshold value
    negatives = (top_confidences*np.logical_not(positives) > threshold_confidences)
    negatives = negatives.astype(np.float32)

    return positives, negatives