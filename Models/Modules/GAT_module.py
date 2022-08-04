# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.Layers.UniGAT_layer import UniformHeaderGraphAttentionLayer
from Models.Layers.GAT_layer import GraphAttentionLayer

class GraphAttentionMoudle(nn.Module):
    
    
    def __init__(self, dim_hid, dim_out, num_heads, alpha, special_name, dropout=0):
        super().__init__()

        self.num_heads = num_heads
        self.dim_out = dim_out

        self.attconv = [UniformHeaderGraphAttentionLayer(dim_hid, dropout=dropout, alpha=alpha, concat=True) for _ in range(self.num_heads)]
        for i, attention in enumerate(self.attconv):
            self.add_module('attconv_'+ special_name + '_{}'.format(i), attention)

        self.attconv_out = [GraphAttentionLayer(dim_hid * self.num_heads, self.dim_out, dropout=dropout, alpha=alpha, concat=True) for _ in range(self.num_heads)]
        for i, attention in enumerate(self.attconv_out):
            self.add_module('attconv_out' + special_name + '_{}'.format(i), attention)

    def forward(self, chems, adj):

        x = torch.cat([self.attconv[i](chems[i], adj) for i in range(self.num_heads)], dim=1)

        x = torch.cat([att(x, adj) for att in self.attconv_out], dim=1)


        return x