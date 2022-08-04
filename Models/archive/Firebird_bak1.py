# -*- coding: utf-8 -*-
"""
Created on Fri Apri 01 19:14:30 2022

@author: Wu Shiauthie
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.Modules.RGCN_module import RelationalGraphConvModel
from Models.Modules.GAT_module import GraphAttentionMoudle
from Models.Modules.WeightGCN_module import WeightedGraphConvolutionModule
from Models.Modules.Swin_module import SwinTransformer
from Models.Modules.Transformer_module import Transformer
from Models.Layers.Logist_layer import LogisticActivation


class FireBird(nn.Module):
    def __init__(self, alpha, dim_DA, dim_DB, dim_mol_props, num_landmarks, dim_in_GCN, dim_hid_GCN, dim_out_GCN, img_size_Swin, patch_size_Swin, dim_inchannel_Swin, window_size_Swin, dim_outchannel_Swin, depths_Swin, num_head_Swin, dim_out_Swin, dim_hid_RGCN, num_bases_RGCN, num_layer_RGCN, dim_out_RGCN, dim_hid_GAT, dim_out_GAT, num_head_GAT, multi_graphs, num_cell_lines, num_layers, num_heads, dim_head, dropout, cuda):
        super().__init__()
        
        self.dim_DB = dim_DB
        self.dropout = dropout
        self.num_landmarks = num_landmarks
        self.num_head_GAT = num_head_GAT

        self.uni_heads = nn.ParameterList([nn.Parameter(torch.empty(size=(dim_out_RGCN+dim_out_Swin, dim_hid_GAT))) for _ in range(num_head_GAT)])

        # self.register_buffer('cell_line_tokens', torch.arange(num_cell_lines).reshape((num_cell_lines, 1)).long()) # using with embedding layer
        self.cls_tokens = nn.Parameter(torch.empty(num_cell_lines, 1, dim_out_GAT*num_head_GAT))

        # GetGeneEmbeddings
        self.landmark_embs = nn.Parameter(torch.empty(num_landmarks, dim_in_GCN))

        # GetExpressGenProfs
        self.weigcn = WeightedGraphConvolutionModule(dim_in=dim_in_GCN, dim_hid=dim_hid_GCN, dim_out=dim_out_GCN, dropout=self.dropout)

        # GetChemProfiles
        self.rgcn = RelationalGraphConvModel(input_size=dim_DA, hidden_size=dim_hid_RGCN, 
                                             output_size=dim_out_RGCN, num_bases=num_bases_RGCN, 
                                             num_rel=dim_DB, num_layer=num_layer_RGCN, dropout=self.dropout)

        # GetConMapProfiles
        self.swin = SwinTransformer(img_size=img_size_Swin, patch_size=patch_size_Swin, 
                                    in_chans=dim_inchannel_Swin, num_classes=dim_out_Swin, embed_dim=dim_outchannel_Swin, depths=depths_Swin, num_heads=num_head_Swin, window_size=6, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm)

        # GetMultiGraphProfiles
        self.multigat_stack = nn.ModuleList()
        
        for graph in multi_graphs:

            self.multigat_stack.append(GraphAttentionMoudle(dim_hid=dim_hid_GAT, dim_out=dim_out_GAT, num_heads=num_head_GAT, alpha=alpha, special_name=graph))
            
            setattr(self, graph, self.multigat_stack[-1])

        # CombineGeneChemcial
        self.trans = Transformer(dim_input=dim_out_GAT*num_head_GAT, dim_hid=2*dim_out_GAT*num_head_GAT, num_layers=num_layers, num_heads=num_heads, dim_head=dim_head)

        # InitDosageScale
        self.dose_scale = LogisticActivation()

        self.reset_parameters()

    def reset_parameters(self):

        for head in self.uni_heads:
            nn.init.xavier_uniform_(head.data, gain=1.414)

        nn.init.xavier_uniform_(self.landmark_embs.data, gain=1.414)
        nn.init.xavier_uniform_(self.cls_tokens.data, gain=1.414)


    def forward(self, bond_feats, atom_feats, mol_props, ConMap, multi_graphs_list, adj_dict, express_gene_adj, weight_gene_relations, data_indices, dose_indices):

        compound_indices = data_indices[:, 0]
        time_indices = data_indices[:, 1]
        cell_line_indices = data_indices[:, 2]

        # weight GCN builds express gene profiles under featureless
        # landmarks = self.weigcn(weight_gene_relations, None, express_gene_adj)

        # RGCN builds chem profiles
        drug_compound = []

        for i in range(bond_feats.shape[0]):
            mols = torch.sum(self.rgcn([chem.squeeze() for chem in bond_feats[i].chunk(self.dim_DB, dim=-1)], atom_feats[i]), dim=0, keepdim=True)
            drug_compound.append(mols)

        rgcn_out = torch.cat(drug_compound, dim=0)

        # SwinTransformer builds chem profiles
        swin_out = self.swin(ConMap)

        # combine chem-chem network graphs inputs
        chem_reps = torch.cat([swin_out, rgcn_out], dim=1)
        # chem_reps = torch.cat([swin_out, rgcn_out, mol_props], dim=1)

        # uniform transformation of chem for next GAT layers
        uniform_h = []

        for head in self.uni_heads:
            h = torch.mm(chem_reps, head)
            assert not torch.isnan(h).any()
            uniform_h.append(h)

        multi_compound_words = []

        for gat in multi_graphs_list:
            multi_compound_words.append(getattr(self, gat)(uniform_h, adj_dict[gat]).unsqueeze(1))

        multi_compound_words = torch.cat(multi_compound_words, dim=1)

        # draw participating chemicals and cell lines from data library
        chem_participate = multi_compound_words[compound_indices]
        cell_line_tokens = self.cls_tokens[cell_line_indices]

        # compute exposed gene expressions
        gene_expressions = self.trans(cell_line_tokens, self.landmark_embs, chem_participate, chem_participate)

        # compute gene expressions affected by compound dosage
        # gene_expressions.shape: (batch, num_genes, dim_gene)
        gene_expressions = torch.mul(gene_expressions, self.dose_scale(dose_indices))

        # compute gene expressions with gene interaction
        gene_expressions = self.weigcn(weight_gene_relations, gene_expressions, express_gene_adj)

        # gene_diff = gene_expressions - self.landmark_embs
        
        return gene_expressions