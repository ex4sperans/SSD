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
        return cls(pd.DataFrame.from_records(boxes,
                                             index=classnames,
                                             columns=BOUNDBOX_COLUMNS))

    def as_centerbox_array(self):
        centerboxes = np.matmul(self.as_matrix(), BOUNDBOX_TO_CENTERBOX)
        return CenterBoxArray.from_boxes(centerboxes, classnames=self.index)


class CenterBoxArray(pd.DataFrame):

    @classmethod
    def from_boxes(cls, boxes, classnames=None):
        return cls(pd.DataFrame.from_records(boxes,
                                             index=classnames,
                                             columns=CENTERBOX_COLUMNS))
    def as_boundbox_array(self):
        boundboxes = np.matmul(self.as_matrix(), CENTERBOX_TO_BOUNDBOX)
        return BoundBoxArray.from_boxes(boundboxes, classnames=self.index)
