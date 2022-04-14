# Model_block.py

import time
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import copy
import math


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        output = self.embedding(x) * math.sqrt(self.d_model)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos = np.expand_dims(np.arange(max_len), 1)
        self.pe = pos / np.power(10000, 2 * np.expand_dims(np.arange(d_model) // 2, 0) / d_model)
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.pe = torch.as_tensor(self.pe).unsqueeze(0)

    def forward(self, x):
        output = x + nn.Parameter(self.pe, requires_grad=False).to('cuda:0')
        output = self.dropout(output)
        return output


class ScaledDotProduct(nn.Module):
    def __init__(self, dropout):
        super(ScaledDotProduct, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        scale = key.size(-1) ** -0.5  # 缩放因子
        attention = torch.matmul(query, key.transpose(-2, -1))
        attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        output = torch.matmul(attention, value)
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = self.w_1(x)
        output = F.relu(output)
        output = self.w_2(output)
        output = self.dropout(output)
        return output


class ResidualConnection(nn.Module):
    def __init__(self, d_model):
        super(ResidualConnection, self).__init__()
        self.d_model = d_model
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, x, sublayer):
        output = x + sublayer
        output = self.layer_norm(output)
        return output