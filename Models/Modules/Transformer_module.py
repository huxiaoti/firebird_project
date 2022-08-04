# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.Layers.Encoder_layer import Encoder

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''
                
    def __init__(self, dim_input, dim_hid, num_layers, num_heads, dim_head, dropout=0.):
        
        super().__init__()

        self.encoder = Encoder(dim_input=dim_input, dim_hid=dim_hid, num_layers=num_layers, num_heads=num_heads, dim_head=dim_head, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, cell_line_emb, src_q_seq, src_k_seq, src_v_seq):

        src_q_seq = torch.cat((cell_line_emb, src_q_seq), dim=1)
        src_q_seq = self.dropout(src_q_seq)

        enc_output = self.encoder(src_q_seq, src_k_seq, src_v_seq)

        output = enc_output[:, 1:]
        

        return output