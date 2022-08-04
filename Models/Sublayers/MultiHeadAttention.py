# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.):
        super().__init__()
        
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):

        attn = torch.matmul(q, k.transpose(2, 3)) * self.temperature
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, num_heads, dim_input, dim_head, dropout=0.):
        super().__init__()

        # n_head = 8
        # d_model = 512
        # d_k = 64
        # d_v = 64
        '''
        n_head = num_heads = heads
        d_model = dim_input = dim
        d_k = d_v = dim_head = dim_head
        inner_dim = dim_head * num_heads
        '''
        
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.inner_dim = num_heads * dim_head
        self.scale = dim_head ** -0.5

        self.w_qs = nn.Linear(dim_input, self.inner_dim, bias=False)
        self.w_ks = nn.Linear(dim_input, self.inner_dim, bias=False)
        self.w_vs = nn.Linear(dim_input, self.inner_dim, bias=False)

        self.fc = nn.Linear(self.inner_dim, dim_input, bias=False)

        self.attention = ScaledDotProductAttention(temperature=self.scale)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_input, eps=1e-6)

    def forward(self, q, k, v):

        batch_size, q_length = q.size(0), q.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(batch_size, -1, self.num_heads, self.dim_head)
        k = self.w_ks(k).view(batch_size, -1, self.num_heads, self.dim_head)
        v = self.w_vs(v).view(batch_size, -1, self.num_heads, self.dim_head)

        # Transpose for attention dot product: batch, num_heads, sentence_length, dim_head
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        out = self.attention(q, k, v)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)

        out = out.transpose(1, 2).contiguous().view(batch_size, q_length, -1)
        out = self.dropout(self.fc(out))
        out = self.layer_norm(out + residual)

        return out