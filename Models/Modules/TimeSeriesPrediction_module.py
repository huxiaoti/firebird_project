# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.Modules import Transformer, WeightedGraphConvolutionModule
from Models.Layers.Logist_layer import LogisticActivation


class TimeSerisePredictionModule(nn.Module):


    def __init__(self, num_genes, dim_out_GAT, num_head_GAT, num_layers_Trans, num_heads_Trans, dim_head_Trans, dim_in_GCN, dim_hid_GCN, dim_out_GCN):
        super().__init__()

        # CombineGeneChemcial
        self.trans = Transformer(dim_input=dim_out_GAT*num_head_GAT, dim_hid=2*dim_out_GAT*num_head_GAT, num_layers=num_layers_Trans, num_heads=num_heads_Trans, dim_head=dim_head_Trans)

        # GetExpressGenProfs
        self.weigcn = WeightedGraphConvolutionModule(dim_in=dim_in_GCN, dim_hid=dim_hid_GCN, dim_out=dim_out_GCN)

        # InitDosageScale
        self.dose_scale = LogisticActivation(num_genes)


    def forward(self, cell_line_tokens, gene_embs, compounds_k, compounds_v, dose_indices, weight_gene_relations, express_gene_adj):

        # compute exposed gene expressions
        gene_latent_feat = self.trans(cell_line_tokens, gene_embs, compounds_k, compounds_v)

        # compute gene expressions affected by compound dosage
        # gene_expressions.shape: (batch, num_genes, dim_gene)
        gene_latent_feat = torch.mul(gene_latent_feat, self.dose_scale(dose_indices))

        # compute gene expressions with gene interaction
        gene_latent_feat = self.weigcn(weight_gene_relations, gene_latent_feat, express_gene_adj)

        return gene_latent_feat