# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedGraphConvolutionLayer(nn.Module):


    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        if self.bias is not None:
            nn.init.xavier_uniform_(self.bias.data, gain=1.414)
        

    def forward(self, weights, feats, adj):

        """
        feats.shape: [batch, num_genes, dim_gene]
        """

        batch_size, num_genes = feats.shape[0], feats.shape[1]

        if feats != None:
            x = torch.matmul(feats, self.weight)
        else:
            x = self.weight
            
        weighted_adj = weights * adj
        feat_aggregation = []
        for sample in x:
            feat_aggregation.append(torch.sparse.mm(weighted_adj, sample))
        output = torch.cat(feat_aggregation, dim=0).reshape(batch_size, num_genes, -1)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'