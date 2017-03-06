import ssd
import boxes
import voc_loader

neg_pos_ratio = 3
overlap_threshold = 0.5
batch_size = 4
n_iter = 100000
learning_rate = 0.0001
test_freq = 1
save_freq = 1

model = ssd.SSD(resume=False)
loader = voc_loader.VOCLoader(
                              preprocessing=('resize', model.input_shape),
                              normalization='divide_255')

model.train(loader, overlap_threshold, neg_pos_ratio,
            batch_size, learning_rate, n_iter, test_freq, save_freq)

