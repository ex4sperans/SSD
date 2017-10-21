import numpy as np
import pandas as pd


BOUNDBOX_COLUMNS = ["x_min", "y_min", "x_max", "y_max"]
CENTERBOX_COLUMNS = ["x_center", "y_center", "width", "height"]

BOUNDBOX_TO_CENTERBOX = np.array([[ 0.5,    0,  -1,    0],
                                  [   0,  0.5,   0,   -1],
                                  [ 0.5,    0,   1,    0],
                                  [   0,  0.5,   0,    1]], dtype=np.float32)

CENTERBOX_TO_BOUNDBOX = np.array([[   1,    0,    1,    0],
                                  [   0,    1,    0,    1],
                                  [-0.5,    0,  0.5,    0],
                                  [   0, -0.5,    0,  0.5]])


class BoundBoxArray(pd.DataFrame):

    @classmethod
    def from_boxes(cls, boxes, classnames=None):
        """Initialize BoundBoxArray from list of bounding boxes.

        Args:
            boxes: list of tuples: (x_min, y_min, x_max, y_max)
            classnames: list of classnames for bounding boxes, optional
        """
        return cls(pd.DataFrame.from_records(boxes,
                                             index=classnames,
                                             columns=BOUNDBOX_COLUMNS))

    def as_centerbox_array(self):
        """Returns correspoding CenterBoxInstance"""
        centerboxes = np.matmul(self.as_matrix(), BOUNDBOX_TO_CENTERBOX)
        return CenterBoxArray.from_boxes(centerboxes, classnames=self.index)

    def rescale(self, scale: tuple):
        """Rescale accoding to `scale`."""

        vertical, horizontal = scale
        scale = np.array([horizontal, vertical, horizontal, vertical])

        scaled = self.as_matrix() / scale
        return BoundBoxArray.from_boxes(scaled, classnames=self.index)


class CenterBoxArray(pd.DataFrame):

    @classmethod
    def from_boxes(cls, boxes, classnames=None):
        """Initialize CenterBoxArray from list of bounding boxes.

        Args:
            boxes: list of tuples: (x_center, y_center, width, height)
            classnames: list of classnames for center boxes, optional
        """
        return cls(pd.DataFrame.from_records(boxes,
                                             index=classnames,
                                             columns=CENTERBOX_COLUMNS))
    def as_boundbox_array(self):
        """Returns correspoding BoundBoxInstance"""
        boundboxes = np.matmul(self.as_matrix(), CENTERBOX_TO_BOUNDBOX)
        return BoundBoxArray.from_boxes(boundboxes, classnames=self.index)

    def rescale(self, scale: tuple):
        """Rescale accoding to `scale`."""

        vertical, horizontal = scale
        scale = np.array([horizontal, vertical, horizontal, vertical])

        scaled = self.as_matrix() / scale
        return CenterBoxArray.from_boxes(scaled, classnames=self.index)
