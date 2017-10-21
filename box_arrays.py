import numpy as np

class BoundBoxArray(np.ndarray):

    def __new__(cls, array):
        return np.asarray(array).view(cls)

    def __init__(self, *args, **kwargs):
    
        if not self.ndim == 2:
            raise ValueError('BoundBoxArray must have the dimension of 2.')
        if not self.shape[1] == 4:
            raise ValueError('BoundBoxArray must have 4 values along second axis.')
        if (self.x_min > self.x_max).any():
            raise ValueError('`x_max` should be greater than `x_min`.')
        if (self.y_min > self.y_max).any():
            raise ValueError('`y_max` should be greater than `y_min`.')

    @property
    def x_min(self):
        return self[:, np.newaxis, 0]

    @property
    def y_min(self):
        return self[:, np.newaxis, 1]

    @property
    def x_max(self):
        return self[:, np.newaxis, 2]

    @property
    def y_max(self):
        return self[:, np.newaxis, 3]

    def as_centerbox_array(self):

        conversion_matrix = np.array([[0.5,   0, 0.5,   0],
                                      [  0, 0.5,   0, 0.5],
                                      [ -1,   0,   1,   0],
                                      [  0,  -1,   0,   1]])

        return CenterBoxArray(np.matmul(self, conversion_matrix.T))

    def as_boundbox_array(self):
        return self

    def normalize(self, height, width):
    
        return BoundBoxArray(np.hstack(
                                       (self.x_min/width,
                                       self.y_min/height,
                                       self.x_max/width,
                                       self.y_max/height)))

class CenterBoxArray(np.ndarray):

    def __new__(cls, array):
        return np.asarray(array).view(cls)

    def __init__(self, *args, **kwargs):
    
        if not self.ndim == 2:
            raise ValueError('BoundBoxArray must have the dimension of 2.')
        if not self.shape[1] == 4:
            raise ValueError('BoundBoxArray must have 4 values along second axis.')

    @property
    def center_x(self):
        return self[:, np.newaxis, 0]

    @property
    def center_y(self):
        return self[:, np.newaxis, 1]

    @property
    def width(self):
        return self[:, np.newaxis, 2]

    @property
    def height(self):
        return self[:, np.newaxis, 3]

    def as_boundbox_array(self):
        conversion_matrix = np.array([[1, 0, -0.5,    0],
                                      [0, 1,    0, -0.5],
                                      [1, 0,  0.5,    0],
                                      [0, 1,    0,  0.5]])
        
        return BoundBoxArray(np.matmul(self, conversion_matrix.T))

    def as_centerbox_array(self):
        return self

    def normalize(self, height, width):
    
        boundboxes = self.as_boundbox_array()
        return BoundBoxArray(np.hstack(
                                       boundboxes.x_min/width,
                                       boundboxes.y_min/height,
                                       boundboxes.x_max/width,
                                       boundboxes.y_max/height)).as_centerbox_array()


if __name__ == '__main__':

    default_boxes = [[1.0, 2.0, 3.0, 8.0 ],
                     [2.0, 5.0, 6.0, 20.0]]

    ground_truth_boxes = [[1.0, 2.0, 3.0, 8.0 ]]

    default_boxes = BoundBoxArray(default_boxes)
    ground_truth_boxes = BoundBoxArray(ground_truth_boxes)
    
    print(default_boxes.jaccard_overlap(ground_truth_boxes))




