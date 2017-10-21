import boxes
import ssd
import voc_loader

neg_pos_ratio = 3
overlap_threshold = 0.5
nms_threshold = 0.45
batch_size = 32
n_iter = 200000
test_freq = 50
save_freq = 1000

def learning_rate_schedule(iteration):

    if iteration < 50000:
        learning_rate = 3e-4
    elif iteration < 100000:
        learning_rate = 3e-5
    else:
        learning_rate = 1e-5
    return learning_rate

model = ssd.SSD(resume=False)
loader = voc_loader.VOCLoader(
                              preprocessing=('resize', model.input_shape),
                              normalization='divide_255',
                              augmentation=[('random_vflip', 0.25),
                                            ('random_hflip', 0.25),
                                            ('random_tile', 0.25),
                                            ('random_crop', 0.25)])

model.train(loader, overlap_threshold, nms_threshold, neg_pos_ratio,
            batch_size, learning_rate_schedule, n_iter, test_freq, save_freq)