# model_transformer.py

import torch
from torch import nn
import torch.nn.functional as F
import copy
import math
import time


class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.config = config

        self.embeddings = nn.Embedding(self.config.vocab_len, self.config.d_embed)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.d_embed, out_channels=self.config.channels_num[0],
                      kernel_size=self.config.kernel_size[0], padding='valid'),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.channels_num[0], out_channels=self.config.channels_num[1],
                      kernel_size=self.config.kernel_size[1], padding='valid'),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.channels_num[1], out_channels=self.config.channels_num[2],
                      kernel_size=self.config.kernel_size[2], padding='valid'),
            nn.ReLU(),
            nn.MaxPool1d(self.config.max_sen_len - self.config.size_num + len(self.config.kernel_size))
        )
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, x):
        embedded = self.embeddings(x).permute(0, 2, 1)

        conv_out1 = self.conv1(embedded)
        conv_out2 = self.conv2(conv_out1)
        conv_out3 = self.conv3(conv_out2)

        output = self.dropout(conv_out3)
        output = output.squeeze()
        return output