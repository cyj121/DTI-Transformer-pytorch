# config_classify.py


class ConfigClassify(object):
    n = 1
    d_model = 96
    d_ff = 192
    h = 8
    dropout = 0.1
    output_size = 2
    batch_size = 256
    fc = [1024, 512]