cdef to_centerbox(float x_min, float y_min, float x_max, float y_max):
    cdef float width = x_max - x_min
    cdef float height = y_max - y_min
    cdef float center_x = x_min + width/2.0
    cdef float center_y = y_min + height/2.0
    return center_x, center_y, width, height

cdef to_boundbox(float center_x, float center_y, float width, float height):
    cdef float half_width = width/2
    cdef float half_height = height/2
    cdef float x_min = center_x - half_width
    cdef float x_max = center_x + half_width
    cdef float y_min = center_y - half_height
    cdef float y_max = center_y + half_height
    return x_min, y_min, x_max, y_max

cdef inline float _intersection(float x_min1, float y_min1,
                                float x_max1, float y_max1,
                                float x_min2, float y_min2,
                                float x_max2, float y_max2):

    cdef float w1 = x_max1 - x_min1
    cdef float h1 = y_max1 - y_min1
    cdef float w2 = x_max2 - x_min2
    cdef float h2 = y_max2 - y_min2

    cdef float intersection_w = (w1 + w2) - (max(x_max1, x_max2) - min(x_min1, x_min2))
    cdef float intersection_h = (h1 + h2) - (max(y_max1, y_max2) - min(y_min1, y_min2))

    return max(intersection_w, 0)*max(intersection_h, 0)

def jaccard_overlap(box1, box2):

    #box1 and box2 are both assumed to be boundboxes 
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    intersection_ = _intersection(x_min1, y_min1, x_max1, y_max1,
                                  x_min2, y_min2, x_max2, y_max2)
    center_x1, center_y1, width1, height1 = to_centerbox(x_min1, y_min1, x_max1, y_max1)
    center_x2, center_y2, width2, height2 = to_centerbox(x_min2, y_min2, x_max2, y_max2)
    union = width1*height1 + width2*height2 - intersection_
    return intersection_/union if union > 0 else 0 
