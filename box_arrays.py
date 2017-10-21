import numpy as np
import pandas as pd

from box_ops import iou


BOUNDBOX_COLUMNS = ["x_min", "y_min", "x_max", "y_max"]
CENTERBOX_COLUMNS = ["x_center", "y_center", "width", "height"]
BOX_COLUMNS = BOUNDBOX_COLUMNS + CENTERBOX_COLUMNS

BOUNDBOX_TO_CENTERBOX = np.array([[ 0.5,    0,  -1,    0],
                                  [   0,  0.5,   0,   -1],
                                  [ 0.5,    0,   1,    0],
                                  [   0,  0.5,   0,    1]], dtype=np.float32)

CENTERBOX_TO_BOUNDBOX = np.array([[   1,    0,    1,    0],
                                  [   0,    1,    0,    1],
                                  [-0.5,    0,  0.5,    0],
                                  [   0, -0.5,    0,  0.5]], dtype=np.float32)


class BoundBoxArray(pd.DataFrame):

    @classmethod
    def from_boxes(cls, boxes, classnames=None):
        """Initialize BoundBoxArray from list or numpy array of boxes.

        Args:
            boxes: list of tuples: (x_min, y_min, x_max, y_max, 
                                    x_center, y_center, width, height),
                or similar numpy array
            classnames: list of classnames for bounding boxes, optional
        """
        boxes = np.array(boxes, dtype=np.float32)

        return cls(pd.DataFrame.from_records(boxes,
                                             index=classnames,
                                             columns=BOX_COLUMNS))

    @classmethod
    def from_boundboxes(cls, boxes, classnames=None):
        """Initialize BoundBoxArray from list or numpy array of bound

        Args:
            boxes: list of tuples: (x_min, y_min, x_max, y_max),
                or similar numpy array
            classnames: list of classnames for bounding boxes, optional
        """
        boundboxes = np.array(boxes, dtype=np.float32)
        centerboxes = np.matmul(boundboxes, BOUNDBOX_TO_CENTERBOX)
        boxes = np.hstack((boundboxes, centerboxes))

        return cls(pd.DataFrame.from_records(boxes,
                                             index=classnames,
                                             columns=BOX_COLUMNS))

    @classmethod
    def from_centerboxes(cls, boxes, classnames=None):
        """Initialize BoundBoxArray from list or numpy array of centerboxes.

        Args:
            boxes: list of tuples: (x_center, y_center, width, height),
                or similar numpy array
            classnames: list of classnames for bounding boxes, optional
        """
        centerboxes = np.array(boxes, dtype=np.float32)
        boundboxes = np.matmul(centerboxes, CENTERBOX_TO_BOUNDBOX)
        boxes = np.hstack((boundboxes, centerboxes))

        return cls(pd.DataFrame
                   .from_records(boxes,
                                 index=classnames,
                                 columns=BOX_COLUMNS))

    def __getattr__(self, attr):
        """Overrides getattr to return np.array rather than pd.Series object"""
        value = pd.DataFrame.__getattr__(self, attr)

        if attr in BOX_COLUMNS:
            return value.as_matrix()
        else:
            return value

    def rescale(self, scale):
        """Rescale accoding to `scale`."""
        vertical, horizontal = scale
        scale = np.array([horizontal, vertical] * 4)

        scaled = self.as_matrix() / scale
        return BoundBoxArray.from_boxes(scaled, classnames=self.index)

    
    def iou(self, other):
        """Compute IOU matrix between current array and other"""
        return iou(self, other)
