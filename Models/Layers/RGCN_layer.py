# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationalGraphConvLayer(nn.Module):
    
    
    def __init__(self, input_size, output_size, num_bases, num_rel, bias=False):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.num_bases = num_bases
        self.num_rel = num_rel

        # R-GCN weights
        if num_bases > 0:
            self.w_bases = nn.Parameter(torch.FloatTensor(self.num_bases, self.input_size, self.output_size))
            self.w_rel = nn.Parameter(torch.FloatTensor(self.num_rel, self.num_bases))
        else:
            self.w = nn.Parameter(torch.FloatTensor(self.num_rel, self.input_size, self.output_size))
        # R-GCN bias
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, self.output_size))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.num_bases > 0:
            nn.init.xavier_uniform_(self.w_bases.data)
            nn.init.xavier_uniform_(self.w_rel.data)
        else:
            nn.init.xavier_uniform_(self.w.data)
        if self.bias is not None:
            nn.init.xavier_uniform_(self.bias.data)

    def forward(self, A, X):

        self.w = (torch.einsum("rb, bio -> rio", (self.w_rel, self.w_bases)) if self.num_bases > 0 else self.w)
        # torch.matmul(b, a.permute(1,0,2))
        weights = self.w.view(self.w.shape[0] * self.w.shape[1], self.w.shape[2])

        # Each relations * Weight
        supports = []
        for i in range(self.num_rel):
            
            supports.append(torch.sparse.mm(A[i], X))

        tmp = torch.cat(supports, dim=1)
        out = torch.mm(tmp, weights)

        if self.bias is not None:
            out = out + self.bias
        return out
