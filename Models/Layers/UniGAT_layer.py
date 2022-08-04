# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class UniformHeaderGraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, out_features, dropout, alpha, concat=True):
        super().__init__()
        self.dropout = dropout
        # self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):

        Wh = h
        # Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)
        assert not torch.isnan(e).any()

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        assert not torch.isnan(h_prime).any()

        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

class SpUniformHeaderGraphAttentionLayer(nn.Module):


    def __init__(self, out_features, dropout, alpha, batch_normal=False, concat=True):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha
        self.concat = concat
        self.batch_normal = batch_normal
        if self.batch_normal:
            self.bn = torch.nn.BatchNorm1d(out_features)

        self.a = nn.Parameter(torch.empty(size=(1, 2*out_features)))
        self.register_buffer('b', torch.ones(size=(1, 1)))

        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.layer_norm = nn.LayerNorm(out_features)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, model_input, adj):


        N = model_input.size()[0]
        edge = adj._indices()

        h = model_input

        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        
        assert not torch.isnan(edge_e).any(), 'Some value is NaN in UniGAT_layer 1.'


        a = torch.sparse_coo_tensor(edge, edge_e, (N, N))
        b = self.b.repeat(N, 1)
        
        e_rowsum = torch.sparse.mm(a, b)

        edge_e = self.dropout(edge_e)

        a = torch.sparse_coo_tensor(edge, edge_e, (N, N))
        
        h_prime = torch.sparse.mm(a, h)
        assert not torch.isnan(h_prime).any(), 'Some value is NaN in UniGAT_layer 2.'

        h_prime = h_prime.div(e_rowsum)
        assert not torch.isnan(h_prime).any(), 'Some value is NaN in UniGAT_layer 3.'
        '''
        h_prime = self.layer_norm(h_prime) # maybe not need or need!!!
        '''
        if self.concat:
            # if this layer is not last layer,
            return F.relu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime