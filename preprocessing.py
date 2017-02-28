import boxes

def offsets(ground_truth_box, default_box):

    ground_truth_box = boxes.boundbox_to_centerbox(ground_truth_box)
    default_box = boxes.boundbox_to_centerbox(default_box)

    return boxes.CenterBox(*[g-d for g, d in zip(ground_truth_box, default_box)])

def process_matches(matches, default_boxes, class_names):

    background_class = len(class_names) + 1

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

