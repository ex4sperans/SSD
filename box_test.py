import boxes
import preprocessing
from matching import match_boxes
from voc_loader import VOCLoader
import misc

ssd_params = misc.load_json('ssd_params.json')
class_names = ssd_params['class_names']

out_shapes = [
              (1, 38, 38, 75),
              (1, 19, 19, 75),
              (1, 10, 10, 75),
              (1, 5, 5, 75),
              (1, 3, 3, 75), 
              (1, 1, 1, 75)]

box_ratios = [[1, 3, 1/3, 1/2, 2]]*6
   
default_boxes = boxes.get_default_boxes(out_shapes, box_ratios)

new_height, new_width = 300, 300
loader = VOCLoader(preprocessing=('resize', (new_height, new_width, 3)),
                   normalization='divide_255')

batch = loader.new_train_batch(1)
image, annotation = batch[0]

objects = annotation['objects']
matches = match_boxes(objects, image, default_boxes, out_shapes, 0.65)
matched_default_boxes = [boxes.recover_box(default_box, new_height, new_width)
                         for default_box, _ in matches]

ground_truth_boxes = [box for class_name, box in objects]

boxes.plot_with_bboxes(
                       image,
                       'plots_with_bbox',
                       annotation['file_name'],
                       matched_default_boxes,
                       ground_truth_boxes)

feed = preprocessing.process_matches(
            matches, default_boxes, class_names)

for offsets, class_label in misc.flatten_list(feed):
    if class_label != len(class_names):
        print(offsets, class_names[class_label])