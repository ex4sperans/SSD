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

def apply_log_offsets(default_box, offsets):

    return boxes.CenterBox(
                           default_box.center_x + offsets.center_x*default_box.width,
                           default_box.center_y + offsets.center_y*default_box.height,
                           default_box.width*np.exp(offsets.width),
                           default_box.height*np.exp(offsets.height))

def non_maximum_supression(confidences, default_boxes, corrections, 
                           class_names, threshold, height, width):

    background_class = len(class_names)
    default_boxes = misc.flatten_list(default_boxes)
    confidences = confidences[:, :-1]
    labels = np.argmax(confidences, 1)
    top_confidences = np.amax(confidences, 1)

    idx = np.flip(np.argsort(top_confidences), 0)
    top_confidences = top_confidences[idx]
    labels = labels[idx]
    default_boxes = [default_boxes[i] for i in idx]
    corrections = corrections[idx]
    non_background_boxes = []

    for (default_box,
        label,
        correction,
        confidence) in zip(
        default_boxes,
        labels,
        corrections,
        top_confidences):

        # if label != background_class:
        correction = boxes.CenterBox(*correction)
        non_background_boxes.append(
            (apply_log_offsets(default_box, correction), label, confidence))
        if confidence < 0.15:
            break

    choices = []

    for corrected_box, label, confidence in non_background_boxes:
        add = True
        for selected_corrected_box, selected_label, _ in choices:
            overlap = boxes.jaccard_overlap(corrected_box, selected_corrected_box)
            if label == selected_label and overlap > threshold:
                add = False
                break
        if add:
            choices.append((corrected_box, label, confidence))
        if len(choices) > 20:
            break

    if choices:
        bboxes, labels, confidences = list(zip(*choices))
        bboxes = list(boxes.clip_box(box) for box in bboxes)
        bboxes = list(boxes.recover_box(box, height, width) for box in bboxes)
        labels = list(class_names[l] for l in labels)
    else:
        bboxes, labels, confidences = [], [], []
    return bboxes, labels, confidences


def draw_top_boxes(batch, confidences, corrections, default_boxes,
                 threshold, save_path, iteration, model):

    height, width = misc.height_and_width(model.input_shape)

    for (box_confidences,
        box_corrections,
        image_annotation_pair) in zip(
        confidences,
        corrections,
        batch):

        image, annotation = image_annotation_pair
        file_name = annotation['file_name']

        bboxes, labels, confidences = non_maximum_supression(
                                                confidences=box_confidences,
                                                default_boxes=default_boxes,
                                                corrections=box_corrections,
                                                class_names=model.class_names,
                                                threshold=threshold,
                                                height=height,
                                                width=width)
        boxes.plot_predicted_bboxes(
                                    image=image,
                                    save_path=save_path,
                                    file_name='iteration{} {}'.format(iteration, file_name),
                                    bboxes=bboxes,
                                    labels=labels,
                                    confidences=confidences,
                                    class_names=model.class_names)
