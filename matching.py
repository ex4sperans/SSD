from math import floor, ceil

import boxes
import misc

def match_boxes(annotations, image, default_boxes, out_shapes, threshold):

    matches = list()
    top_match = {'overlap': 0, 'match': (None, None, None)}
    image_height, image_width = boxes.height_and_width(image.shape)
    #default boxes is assumed to be a 4d nested list
    for class_name, ground_truth_box in annotations:
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
                        if overlap > top_match['overlap']:
                            top_match['overlap'] = overlap
                            top_match['match'] = (default_box, (scaled_ground_truth_box, class_name))

    return matches, [top_match['match']]    