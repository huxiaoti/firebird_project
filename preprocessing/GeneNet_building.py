# -*- coding: utf-8 -*-

import numpy as np
import pickle as pkl
import scipy.sparse as sp
import networkx as nx
from community import community_louvain
from networkx.algorithms.community import greedy_modularity_communities
from Models.Utils import idx_map_convert

with open('data/raw_data/HumanNet-FN.tsv', 'r') as f:
    lines = f.readlines()

GeneNet_list = []
GeneName_list = []
for line in lines:

    line = line.strip().split('\t')
    
    GeneNet_list.append([line[0], line[1]])

    if line[0] not in GeneName_list:
        GeneName_list.append(line[0])
    
    if line[1] not in GeneName_list:
        GeneName_list.append(line[1])

GeneNet_list = np.array(GeneNet_list)

idx_map = idx_map_convert(GeneName_list)

G = nx.Graph()

G.add_nodes_from(range(len(idx_map)))
edges = np.array(list(map(idx_map.get, GeneNet_list.flatten())), dtype=np.int32).reshape(GeneNet_list.shape)

G.add_edges_from(edges)

communities = greedy_modularity_communities(G)

partition = community_louvain.best_partition(G)

Gene2Comm = {}
for gene_idx in range(len(idx_map)):

    Gene2Comm[gene_idx] = []

    for comm_idx in range(len(communities)):
        if gene_idx in communities[comm_idx]:
            Gene2Comm[gene_idx].append(comm_idx)

with open('data/raw_data/Gene2Comm_GreedyModularity.pkl','wb') as fw:
    pkl.dump(Gene2Comm, fw)

with open('data/raw_data/Gene2Comm_LouvainCommunity.pkl','wb') as fw:
    pkl.dump(partition, fw)

with open('data/raw_data/GreedyModularity_profiles.pkl','wb') as fw:
    pkl.dump(communities, fw)

with open('data/raw_data/LouvainCommunity_profiles.pkl','wb') as fw:
    pkl.dump(partition, fw)

with open('data/processed_data/gene_idx_map.pkl','wb') as fw:
    pkl.dump(idx_map, fw)