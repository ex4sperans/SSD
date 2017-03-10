import ssd
import boxes
import voc_loader

neg_pos_ratio = 3
overlap_threshold = 0.65
nms_threshold = 0.3
batch_size = 4
n_iter = 100000
test_freq = 1
save_freq = 1

def learning_rate_schedule(iteration):

    if iteration < 100:
        learning_rate = 1e-4
    elif iteration < 20000:
        learning_rate = 1e-3
    elif iteration < 30000:
        learning_rate = 1e-4
    else:
        learning_rate = 1e-5
    return learning_rate

model = ssd.SSD(resume=True)
loader = voc_loader.VOCLoader(
                              preprocessing=('resize', model.input_shape),
                              normalization='divide_255')

model.train(loader, overlap_threshold, nms_threshold, neg_pos_ratio,
            batch_size, learning_rate_schedule, n_iter, test_freq, save_freq)