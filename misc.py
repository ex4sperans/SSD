import itertools

def flatten_list(nested_list):
    #flatten nested list of arbitrary depth
    if isinstance(nested_list[0], list):
        return flatten_list(list(itertools.chain.from_iterable(nested_list)))
    else:
        return nested_list

def height_and_width(shape):
    if len(shape) == 4:
        return shape[1], shape[2]
    elif len(shape) == 3:
        return shape[0], shape[1]
    else:
        raise ValueError('Could not infer height and'\
            ' width from shape {shape}.'.format(shape=shape))
