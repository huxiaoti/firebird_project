# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from Models.Sublayers import MultiHeadAttention, PositionwiseFeedForward

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, dim_input, dim_hid, num_heads, dim_head, dropout=0.):
        super().__init__()

        self.slf_attn = MultiHeadAttention(num_heads, dim_input, dim_head, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(dim_input, dim_hid, dropout=dropout)

    def forward(self, enc_q_input, enc_k_input, enc_v_input):
        
        enc_output = self.slf_attn(enc_q_input, enc_k_input, enc_v_input)
        enc_output = self.pos_ffn(enc_output)

        return enc_output

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, num_layers, num_heads, dim_head, dim_input, dim_hid, dropout=0.):

            super().__init__()

            self.layer_stack = nn.ModuleList([
                EncoderLayer(dim_input, dim_hid, num_heads, dim_head, dropout=dropout)
                for _ in range(num_layers)])

            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(dim_input, eps=1e-6)


    def forward(self, src_q_seq, src_k_seq, src_v_seq):

        enc_q_seq = self.layer_norm(self.dropout(src_q_seq))
        enc_k_seq = self.layer_norm(self.dropout(src_k_seq))
        enc_v_seq = self.layer_norm(self.dropout(src_v_seq))

        for enc_layer in self.layer_stack:

            enc_q_seq = enc_layer(enc_q_seq, enc_k_seq, enc_v_seq)

        return enc_q_seq