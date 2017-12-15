import numpy as np
import pandas as pd

from ops.box_ops import iou


class BoundBoxArray(pd.DataFrame):

    BOUNDBOX_COLUMNS = ["x_min", "y_min", "x_max", "y_max"]
    CENTERBOX_COLUMNS = ["x_center", "y_center", "width", "height"]
    BOX_COLUMNS = BOUNDBOX_COLUMNS + CENTERBOX_COLUMNS

    BOUNDBOX_TO_CENTERBOX = np.array([[ 0.5,    0,  -1,    0],
                                      [   0,  0.5,   0,   -1],
                                      [ 0.5,    0,   1,    0],
                                      [   0,  0.5,   0,    1]],
                                     dtype=np.float32)

    CENTERBOX_TO_BOUNDBOX = np.array([[   1,    0,    1,    0],
                                      [   0,    1,    0,    1],
                                      [-0.5,    0,  0.5,    0],
                                      [   0, -0.5,    0,  0.5]],
                                     dtype=np.float32)

    @property
    def classnames(self):
        if any(not isinstance(i, str) for i in self.index):
            raise AttributeError("Current BoundBoxArray index doesn't"
                                 "represent boxes classnames. Use obj.index "
                                 "to get explicitly get index.")
        return self.index

    @property
    def boundboxes(self):
        return self[self.__class__.BOUNDBOX_COLUMNS]

    @property
    def centerboxes(self):
        return self[self.__class__.CENTERBOX_COLUMNS]

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

        return cls(boxes,
                   index=classnames,
                   columns=cls.BOX_COLUMNS,
                   dtype=np.float32)

    @classmethod
    def from_boundboxes(cls, boxes, classnames=None):
        """Initialize BoundBoxArray from list or numpy array of bound

        Args:
            boxes: list of tuples: (x_min, y_min, x_max, y_max),
                or similar numpy array
            classnames: list of classnames for bounding boxes, optional
        """
        boundboxes = np.array(boxes, dtype=np.float32)
        centerboxes = np.matmul(boundboxes, cls.BOUNDBOX_TO_CENTERBOX)
        boxes = np.hstack((boundboxes, centerboxes))

        return cls(boxes,
                   index=classnames,
                   columns=cls.BOX_COLUMNS,
                   dtype=np.float32)

    @classmethod
    def from_centerboxes(cls, boxes, classnames=None):
        """Initialize BoundBoxArray from list or numpy array of centerboxes.

        Args:
            boxes: list of tuples: (x_center, y_center, width, height),
                or similar numpy array
            classnames: list of classnames for bounding boxes, optional
        """
        centerboxes = np.array(boxes, dtype=np.float32)
        boundboxes = np.matmul(centerboxes, cls.CENTERBOX_TO_BOUNDBOX)
        boxes = np.hstack((boundboxes, centerboxes))

        return cls(boxes,
                   index=classnames,
                   columns=cls.BOX_COLUMNS,
                   dtype=np.float32)

    def __getattr__(self, attr):
        """Overrides __getattribute__ to return
        np.array rather than pd.Series object"""
        value = pd.DataFrame.__getattr__(self, attr)

        if attr in self.__class__.BOX_COLUMNS:
            return value.as_matrix()
        else:
            return value
    
    def __getitem__(self, key):
        """Overrides __getitem__ to return BoundBoxArray
        rather than pd.DataFrame"""
        data = pd.DataFrame.__getitem__(self, key)
        if isinstance(data, pd.DataFrame):
            return BoundBoxArray(data)
        else:
            return data

    def rescale(self, scale):
        """Rescale accoding to `scale`."""
        vertical, horizontal = scale
        scale = np.array([horizontal, vertical] * 4)

        scaled = self.as_matrix() / scale
        return BoundBoxArray.from_boxes(scaled, classnames=self.index)

    def clip(self,
             vertical_clip_value=(0, 1),
             horizontal_clip_value=(0, 1)):
        """Clips all bounding box to lie in range of `vertical_clip_value`
            for y axis and `horizontal_clip_value` for x axis"""

        min_vertical, max_vertical = vertical_clip_value
        min_horizontal, max_horizontal = horizontal_clip_value

        x_min = self.x_min.clip(min=min_horizontal)
        y_min = self.y_min.clip(min=min_vertical)
        x_max = self.x_max.clip(max=max_horizontal)
        y_max = self.y_max.clip(max=max_vertical)

        clipped = np.transpose((x_min, y_min, x_max, y_max))

        return BoundBoxArray.from_boundboxes(clipped, classnames=self.index)

    def iou(self, other):
        """Compute IOU matrix between current array and other"""
        return iou(self, other)

    