# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpressPredictionModule(nn.Module):

    def __init__(self, dim_in, dim_hid, dim_out):
        super().__init__()

        self.express_latent = nn.Linear(dim_in, dim_hid, bias=True)
        self.express_pred = nn.Linear(dim_hid, dim_out, bias=True)
        self.relu = nn.ReLU()

        self.reset_parameter()

    def reset_parameter(self):

        gain_relu = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.express_latent.weight, gain=gain_relu)
        nn.init.xavier_uniform_(self.express_pred.weight, gain=gain_relu)

        nn.init.constant_(self.express_latent.bias, 0)
        nn.init.constant_(self.express_pred.bias, 0)


    def forward(self, gene_latent_feat_archive, gene_latent_feat_forward):

        # difference of gene expression
        minus_expression = gene_latent_feat_forward - gene_latent_feat_archive

        # multiplication of gene expression
        multi_expression = gene_latent_feat_forward * gene_latent_feat_archive

        # concatenation of gene profiles
        gene_latent_feat = torch.cat([gene_latent_feat_forward, minus_expression, multi_expression], dim=-1)

        gene_latent_feat = self.express_latent(gene_latent_feat)

        gene_latent_feat = self.relu(gene_latent_feat)

        gene_expression = self.express_pred(gene_latent_feat) # (batch, gene, 1)

        gene_expression = gene_expression.squeeze(-1) # (batch, gene)

        return gene_expression