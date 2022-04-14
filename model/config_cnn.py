# config.py


class ConfigCNN(object):
    d_embed = 128
    channels_num = [32, 32*2, 32*3]
    kernel_size = [4, 8, 12]
    size_num = kernel_size[0] + kernel_size[1] + kernel_size[2]
    lr = 0.001
    batch_size = 256
    max_sen_len = 700
    dropout = 0.1
    vocab_len = 62