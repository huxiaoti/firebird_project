# -*- coding: utf-8 -*-

import numpy as np
import pickle as pkl
import scipy.sparse as sp
import networkx as nx
from Models.Utils import idx_map_convert, list2graph, normalize_adj, vec2tensor


LandMarks_GeneID_list = []
with open('data/raw_data/GraphRepur_drug_exposure_gene_expression_profiles.csv','r') as f:
    line = f.readline()
    line = f.readline()
    while line:
        GeneID = line.split(',')[0].strip()
        LandMarks_GeneID_list.append(GeneID)
        line = f.readline()

with open('data/raw_data/HumanNet-FN.tsv', 'r') as f:
    lines = f.readlines()

HumanNet_dict = {}
for line in lines:
    line = line.split('\t')
    HumanNet_dict[line[0] + '\t' + line[1]] = line[2][:5].strip()

idx_map_LandMarks = idx_map_convert(LandMarks_GeneID_list)

gene_gene_values_dict = {}
gene_gene_interaction_list = []
value_list = []

for k in HumanNet_dict.keys():
    k_list = k.split('\t')
    if k_list[0] in LandMarks_GeneID_list and k_list[1] in LandMarks_GeneID_list:
        gene_gene_values_dict[str(idx_map_LandMarks[k_list[0]]) + '\t' + str(idx_map_LandMarks[k_list[1]])] = float(HumanNet_dict[k])
        value_list.append(float(HumanNet_dict[k]))
        gene_gene_interaction_list.append([k_list[0], k_list[1]])

max_value = max(value_list)
min_value = min(value_list)
distance = max_value - min_value

for k in gene_gene_values_dict.keys():
    gene_gene_values_dict[k] = (gene_gene_values_dict[k] - min_value)/distance
        
egde_array = np.array(gene_gene_interaction_list)
edges = np.array(list(map(idx_map_LandMarks.get, egde_array.flatten())), dtype=np.int32).reshape(egde_array.shape)
adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(len(idx_map_LandMarks), len(idx_map_LandMarks)), dtype=np.float32)
adj_dense = adj.toarray()
adj = adj + adj.T
adj = adj + sp.eye(adj.shape[0])
adj[adj > 1] = 1
adj = normalize_adj(adj)
adj = adj.tocoo()
adj = vec2tensor(adj)

with open('data/processed_data/Gene_Asso.pkl','wb') as fw:
    pkl.dump(adj, fw)

WeightMatrix = np.zeros((adj.shape[0], adj.shape[1]))

for i in range(adj.shape[0]):
    for j in range(adj.shape[1]):
        if adj_dense[i, j] != 0:
            WeightMatrix[i, j] = gene_gene_values_dict[str(i) + '\t' + str(j)]

WeightMatrix = sp.coo_matrix(WeightMatrix)
WeightMatrix = WeightMatrix + WeightMatrix.T
WeightMatrix = WeightMatrix + sp.eye(WeightMatrix.shape[0])
WeightMatrix = WeightMatrix.tocoo()
WeightMatrix = vec2tensor(WeightMatrix)

with open('data/processed_data/Weighted_Gene_Asso.pkl','wb') as fw:
    pkl.dump(WeightMatrix, fw)