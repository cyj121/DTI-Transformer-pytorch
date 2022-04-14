# model_encoder.py

import torch
import copy
from torch import nn
from model.model_block import *
from model.model_struct import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, h, d_model, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.h = h
        self.d_model = d_model
        self.d_ff = d_ff
        self.multi_head_attention = MultiHeadAttention(self.h, self.d_model, dropout)
        self.position_wise_feed_forward = PositionwiseFeedForward(self.d_model, self.d_ff, dropout)
        self.residual_connection = ResidualConnection(self.d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        sublayer_1 = self.residual_connection(value, self.multi_head_attention(query, key, value))
        sublayer_2 = self.residual_connection(sublayer_1, self.position_wise_feed_forward(sublayer_1))
        return sublayer_2


class Encoder1(nn.Module):
    def __init__(self, config):
        super(Encoder1, self).__init__()
        self.config = config
        self.emdedding = Embeddings(self.config.d_model, self.config.vocab_len)
        self.pisition_embedding = PositionalEncoding(self.config.d_model, self.config.dropout, self.config.max_sen_len)
        self.encoder_layer = EncoderLayer(self.config.h, self.config.d_model, self.config.d_ff, self.config.dropout)
        self.encoder = nn.ModuleList([copy.deepcopy(self.encoder_layer) for _ in range(self.config.n)])
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=self.config.d_model, out_channels=self.config.channels_num,
                      kernel_size=self.config.kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(self.config.max_sen_len - self.config.kernel_size + 1)
        )

    def forward(self, x):
        context = self.emdedding(x)
        context = self.pisition_embedding(context)
        output = context.to(torch.float32)
        for encoder_layer in self.encoder:
            output = encoder_layer(output, output, output)
        output = output.permute(0, 2, 1)
        output = self.conv(output)
        output = output.squeeze()
        return output


class Encoder2(nn.Module):
    def __init__(self, config):
        super(Encoder2, self).__init__()
        self.config = config
        self.encoder_layer = EncoderLayer(self.config.h, self.config.d_model, self.config.d_ff, self.config.dropout)
        self.encoder = nn.ModuleList([copy.deepcopy(self.encoder_layer) for _ in range(self.config.n)])

    def forward(self, query, key, value):
        for encoder_layer in self.encoder:
            output = encoder_layer(query, key, value)
        return output