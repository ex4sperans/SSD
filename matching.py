from math import floor, ceil, log
import numpy as np
import boxes
import misc


def log_offsets(ground_truth_box, default_box):

    #offsets based on ratios
    ground_truth_box = boxes.boundbox_to_centerbox(ground_truth_box)
    default_box = boxes.boundbox_to_centerbox(default_box)

    return boxes.CenterBox(
                           (ground_truth_box.center_x - default_box.center_x)/default_box.width,
                           (ground_truth_box.center_y - default_box.center_y)/default_box.height,
                           log(ground_truth_box.width/default_box.width),
                           log(ground_truth_box.height/default_box.height))

def match_with_offsets(annotations, image, default_boxes, out_shapes, class_names, threshold):

    matches = list()
    background_class = len(class_names)
    image_height, image_width = boxes.height_and_width(image.shape)
    negative_match = match = (boxes.CenterBox(0, 0, 0, 0), background_class)
    if not len(annotations):
        return [negative_match]*len(misc.flatten_list(default_boxes))
    #default boxes is assumed to be a 4d nested list
    for class_name, ground_truth_box in annotations:
        top_match = {'overlap': 0, 'match': None}
        matched_boxes = 0
        box_counter = 0
        for i, out_shape in enumerate(out_shapes):
            out_height, out_width = misc.height_and_width(out_shape)
            ground_truth_box = boxes.centerbox_to_boundbox(ground_truth_box)
            scaled_ground_truth_box = boxes.normalize_box(ground_truth_box, image_height, image_width)
            
            x_min = max(floor(scaled_ground_truth_box.x_min*out_width), 0)
            y_min = max(floor(scaled_ground_truth_box.y_min*out_height), 0)
            x_max = min(ceil((scaled_ground_truth_box.x_max)*out_width), out_width)
            y_max = min(ceil((scaled_ground_truth_box.y_max)*out_height), out_height)

            for x in range(out_width):
                for y in range(out_height):
                    for default_box in default_boxes[i][x][y]:
                        match = negative_match
                        if x_min < x < x_max and y_min < y < y_max:
                            overlap = boxes.jaccard_overlap(scaled_ground_truth_box, default_box)
                            if overlap > threshold:
                                offsets = log_offsets(scaled_ground_truth_box, default_box)
                                match = (offsets, class_names.index(class_name))
                                matched_boxes += 1
                                    
                            if overlap > top_match['overlap']:
                                offsets = log_offsets(scaled_ground_truth_box, default_box)
                                top_match['overlap'] = overlap
                                top_match['match'] = (offsets, class_names.index(class_name)) 
                                top_match['box_counter'] = box_counter
                                top_match['default_box'] = default_box
                        if len(matches) > box_counter:
                            matches[box_counter] = match
                        else:
                            matches.append(match)
                        box_counter += 1

        # if ground truth box was not matched to any default box
        # choose top match as only matched box
        if not matched_boxes:
            if top_match['match'] is not None:
                matches[top_match['box_counter']] = top_match['match']

    return matches


def match_boxes(annotations, image, default_boxes, out_shapes, threshold):

    matches = list()
    image_height, image_width = boxes.height_and_width(image.shape)
    #default boxes is assumed to be a 4d nested list
    for class_name, ground_truth_box in annotations:
        top_match = {'overlap': 0, 'match': None}
        matched_boxes = 0
        for i, out_shape in enumerate(out_shapes):
            out_height, out_width = misc.height_and_width(out_shape)
            ground_truth_box = boxes.centerbox_to_boundbox(ground_truth_box)
            scaled_ground_truth_box = boxes.normalize_box(ground_truth_box, image_height, image_width)
            
            x_min = max(floor(scaled_ground_truth_box.x_min*out_width), 0)
            y_min = max(floor(scaled_ground_truth_box.y_min*out_height), 0)
            x_max = min(ceil((scaled_ground_truth_box.x_max)*out_width), out_width)
            y_max = min(ceil((scaled_ground_truth_box.y_max)*out_height), out_height)

            for x in range(x_min, x_max):
                for y in range(y_min, y_max):
                    for default_box in default_boxes[i][x][y]:
                        overlap = boxes.jaccard_overlap(scaled_ground_truth_box, default_box)
                        if overlap > threshold:
                            matches.append((default_box, (scaled_ground_truth_box, class_name)))
                            matched_boxes += 1
                        if overlap > top_match['overlap']:
                            top_match['overlap'] = overlap
                            top_match['match'] = (default_box, (scaled_ground_truth_box, class_name))

        # if ground truth boxes was not matched to any default box
        # choose top match as only matched box
        if not matched_boxes:
            if top_match['match'] is not None:
                matches.append(top_match['match'])

    return matches