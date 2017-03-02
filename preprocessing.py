import boxes
import misc
from matching import match_boxes

def offsets(ground_truth_box, default_box):

    ground_truth_box = boxes.boundbox_to_centerbox(ground_truth_box)
    default_box = boxes.boundbox_to_centerbox(default_box)

    return boxes.CenterBox(
                           ground_truth_box.center_x - default_box.center_x,
                           ground_truth_box.center_y - default_box.center_y,
                           ground_truth_box.width - default_box.width,
                           ground_truth_box.width - default_box.width)

def process_matches(matches, default_boxes, class_names):

    background_class = len(class_names)

    def offsets_and_class(default_box):
        for matched_default_box, (ground_truth_box, class_name) in matches:
            if default_box == matched_default_box:
                return offsets(ground_truth_box, default_box), class_names.index(class_name)
        return boxes.CenterBox(0, 0, 0, 0), background_class

    feed = [[[[offsets_and_class(default_box)
               for default_box in y_boxes] 
               for y_boxes in x_boxes]
               for x_boxes in out_boxes]
               for out_boxes in default_boxes]

    return feed

def get_feed(batch, model, default_boxes, threshold):
    images = [image for image, annotation in batch]
    matches_batch = [match_boxes(annotations['objects'], image, default_boxes, model.out_shapes, threshold)[0]
                    for image, annotations in batch]
    feed = [misc.flatten_list(process_matches(matches, default_boxes, model.class_names))
                 for matches in matches_batch]

    #split feed batch into offsets and labels batches
    offsets, labels = list(zip(*[list(zip(*f)) for f in feed]))
    return images, list(offsets), list(labels)