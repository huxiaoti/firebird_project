# -*- coding: utf-8 -*-
"""
Created on Fri Apri 01 19:14:30 2022

@author: Wu Shiauthie
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.Modules import RelationalGraphConvModel, GraphAttentionMoudle,  SwinTransformer, TimeSerisePredictionModule, ExpressPredictionModule


class FireBird(nn.Module):
    def __init__(self, alpha, dim_DA, dim_DB, dim_mol_props, num_landmarks, dim_hid_GCN, img_size_Swin, patch_size_Swin, dim_inchannel_Swin, window_size_Swin, dim_outchannel_Swin, depths_Swin, num_head_Swin, dim_out_Swin, dim_hid_RGCN, num_bases_RGCN, num_layer_RGCN, dim_out_RGCN, dim_hid_GAT, dim_out_GAT, num_head_GAT, multi_graphs, num_cell_lines, num_layers_Trans, num_heads_Trans, dim_head_Trans, dosing_time_lib, dropout, cuda):
        super().__init__()
        
        self.dim_DB = dim_DB
        self.dropout = dropout
        self.num_landmarks = num_landmarks
        self.num_head_GAT = num_head_GAT
        self.dosing_time_lib = dosing_time_lib

        self.uni_heads = nn.ParameterList([nn.Parameter(torch.empty(size=(dim_out_RGCN+dim_out_Swin, dim_hid_GAT))) for _ in range(num_head_GAT)])

        # self.register_buffer('cell_line_tokens', torch.arange(num_cell_lines).reshape((num_cell_lines, 1)).long()) # using with embedding layer
        self.cls_tokens = nn.Parameter(torch.empty(num_cell_lines, 1, dim_out_GAT*num_head_GAT))

        # GetGeneEmbeddings
        self.landmark_embs = nn.Parameter(torch.empty(num_landmarks, dim_out_GAT*num_head_GAT))

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

        # GetTimeSeriesLatentInformation
        self.time_serise = TimeSerisePredictionModule(num_genes=num_landmarks, dim_out_GAT=dim_out_GAT, num_head_GAT=num_head_GAT, num_layers_Trans=num_layers_Trans, num_heads_Trans=num_heads_Trans, dim_head_Trans=dim_head_Trans, dim_in_GCN=dim_out_GAT*num_head_GAT, dim_hid_GCN=dim_hid_GCN, dim_out_GCN=dim_out_GAT*num_head_GAT)

        # CalculateGeneExpress
        self.express_pred = ExpressPredictionModule(dim_in=3*dim_out_GAT*num_head_GAT, dim_hid=dim_out_GAT*num_head_GAT*2, dim_out=1)

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
        batch_size = chem_participate.shape[0]
        cell_line_tokens = self.cls_tokens[cell_line_indices]

        # compute time series gene latent features
        gene_feature_stages = []
        for i in range(len(self.dosing_time_lib)+1):

            if i == 0:
                gene_latent_feat = self.landmark_embs.unsqueeze(0)
                gene_latent_feat = gene_latent_feat.repeat(batch_size, 1, 1)

            gene_latent_feat = self.time_serise(cell_line_tokens, gene_latent_feat, chem_participate, chem_participate, dose_indices, weight_gene_relations, express_gene_adj)

            # if i != 0:
            gene_feature_stages.append(gene_latent_feat)

        # comput time series gene expressions
        gene_expression_stages = []
        for stage_idx in range(1, len(gene_feature_stages)):

            gene_expression = self.express_pred(gene_feature_stages[stage_idx-1], gene_feature_stages[stage_idx])

            gene_expression_stages.append(gene_expression)
        
        gene_expression_outputs = []
        for i, k in enumerate(time_indices):

            # k = k - 1
            gene_expression_outputs.append(gene_expression_stages[k][i].unsqueeze(0))

        gene_expression_outputs = torch.cat(gene_expression_outputs, dim=0)

        return gene_expression_outputs