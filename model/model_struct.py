# model_struct.py

import torch
from torch import nn
from model.model_block import *
import torch.nn.functional as F
import copy
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.h = h
        self.d_model = d_model
        self.d_k = self.d_model // self.h
        self.fc_Q = nn.Linear(self.d_model, self.d_model)
        self.fc_K = nn.Linear(self.d_model, self.d_model)
        self.fc_V = nn.Linear(self.d_model, self.d_model)
        self.attention = ScaledDotProduct(dropout)
        self.fc = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.fc_Q(query)
        K = self.fc_K(key)
        V = self.fc_V(value)  # [128, 60, 256]
        Q = Q.view(batch_size, -1, self.h, self.d_k).transpose(-3, -2)
        K = K.view(batch_size, -1, self.h, self.d_k).transpose(-3, -2)
        V = V.view(batch_size, -1, self.h, self.d_k).transpose(-3, -2)
        context = self.attention(Q, K, V)
        context = context.transpose(-3, -2).contiguous().view(batch_size, -1, self.h * self.d_k)
        output = self.fc(context)
        output = self.dropout(output)
        return output