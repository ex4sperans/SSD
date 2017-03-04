import ssd
import boxes
import voc_loader

neg_pos_ratio = 3
overlap_threshold = 0.5
batch_size = 8
n_iter = 100
learning_rate = 0.001
test_freq = 50

model = ssd.SSD()
loader = voc_loader.VOCLoader(
                              preprocessing=('resize', model.input_shape),
                              normalization='divide_255')

model.train(loader, overlap_threshold, neg_pos_ratio,
            batch_size, learning_rate, n_iter, test_freq)