# -*- coding: utf-8 -*-

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, dim_input, dim_hid, dropout=0.):
        super().__init__()
        self.w_1 = nn.Linear(dim_input, dim_hid)
        self.w_2 = nn.Linear(dim_hid, dim_input)

        self.activation =  nn.GELU()

        self.layer_norm = nn.LayerNorm(dim_input, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        out = self.activation(self.w_1(x))
        out = self.dropout(out)
        out = self.w_2(out)
        out = self.dropout(out)

        out = self.layer_norm(out + residual)

        return out