import numpy as np
import matplotlib.pyplot as plt

import misc
import boxes

def apply_offsets(default_box, offsets):

    return boxes.CenterBox(
                           default_box.center_x + offsets.center_x,
                           default_box.center_y + offsets.center_y,
                           default_box.width + offsets.width,
                           default_box.height + offsets.height)

def non_maximum_supression(confidences, default_boxes, corrections, 
                           class_names, threshold, height, width):

    background_class = len(class_names)
    default_boxes = misc.flatten_list(default_boxes)
    labels = np.argmax(confidences, 1)
    top_confidences = np.amax(confidences, 1)
    idx = np.flip(np.argsort(top_confidences), 0)

    top_confidences = top_confidences[idx] 
    labels = labels[idx]
    default_boxes = [default_boxes[i] for i in idx]
    corrections = corrections[idx]
    non_background_boxes = []

    for default_box, label, correction, confidence in zip(default_boxes, labels, corrections, top_confidences):
        if label != background_class:
            correction = boxes.CenterBox(*correction)
            non_background_boxes.append((apply_offsets(default_box, correction), label))
        if confidence < 0.05:
            break

    choices = []

    add = True
    for corrected_box, label in non_background_boxes:
        for selected_corrected_box, selected_label in choices:
            overlap = boxes.jaccard_overlap(corrected_box, selected_corrected_box)
            if label == selected_label and overlap > threshold:
                add = False
                break
        if add:
            choices.append((corrected_box, label))
        if len(choices) > 50: 
            break

    bboxes, labels = list(zip(*choices))
    bboxes = list(boxes.recover_box(box, height, width) for box in bboxes)
    labels = list(class_names[l] for l in labels)
    return bboxes, labels


def draw_top_boxes(image, confidences, corrections, default_boxes,
                 threshold, save_path, file_name, model):

    height, width = misc.height_and_width(model.input_shape)
    bboxes, labels = non_maximum_supression(
                                            confidences,
                                            default_boxes,
                                            corrections,
                                            model.class_names,
                                            threshold,
                                            height,
                                            width)
    boxes.plot_predicted_bboxes(
                                image,
                                save_path,
                                file_name,
                                bboxes=bboxes,
                                labels=labels)