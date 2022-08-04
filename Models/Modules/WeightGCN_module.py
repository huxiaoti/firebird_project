# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.Layers.WeightGCN_layer import WeightedGraphConvolutionLayer

class WeightedGraphConvolutionModule(nn.Module):


    def __init__(self, dim_in, dim_hid, dim_out, dropout=0.):
        super().__init__()

        self.gc1 = WeightedGraphConvolutionLayer(dim_in, dim_hid)
        self.gc2 = WeightedGraphConvolutionLayer(dim_hid, dim_out)
        self.dropout = dropout

    def forward(self, weights, feats, adj):

        x = F.relu(self.gc1(weights, feats, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        output = F.relu(self.gc2(weights, x, adj))

        return output