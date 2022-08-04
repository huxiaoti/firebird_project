# -*- coding: utf-8 -*-

import yaml
import pickle as pkl
import scipy.sparse as sp
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from Models.Utils import idx_map_convert, list2graph, normalize_adj, vec2tensor, detection_association

with open('data/raw_data/chem_web.pkl', 'rb') as fb:
    cid_dict = pkl.load(fb)

with open('data/processed_data/configurations.yml', 'r') as fr:
    params = yaml.load(fr, Loader=yaml.FullLoader)

with open('data/processed_data/gene_idx_map.pkl', 'rb') as fb:
    gene_idx_map = pkl.load(fb)
    
with open('data/raw_data/Gene2Comm_GreedyModularity.pkl', 'rb') as fb:
    Gene2Comm = pkl.load(fb)

chem_CTodbGene_dict = {}

for key_1 in cid_dict.keys():
    chem_CTodbGene_dict[key_1] = []
    for key_2 in cid_dict[key_1]['CTD_chem_gene']:
        if str(key_2) in gene_idx_map.keys():
            chem_CTodbGene_dict[key_1].append(gene_idx_map[str(key_2)])

chem_CTodbComm_dict = {}

for key_1 in chem_CTodbGene_dict.keys():
    chem_CTodbComm_dict[key_1] = []
    for key_2 in chem_CTodbGene_dict[key_1]:
        chem_CTodbComm_dict[key_1].append(Gene2Comm[key_2][0]) # as there are at most one element in each Gene2Comm item


chem_CTodbGene_interaction_list = []

for key_1 in chem_CTodbComm_dict:
    for key_2 in chem_CTodbComm_dict:
        if detection_association(chem_CTodbComm_dict[key_1], chem_CTodbComm_dict[key_2], params['threshold']):
            """
                threshold = 1: 701772 edges
                threshold = 2: 560966 edges
                threshold = 3: 500921 edges
            """
            chem_CTodbGene_interaction_list.append([key_1, key_2])

chem_idx_map = idx_map_convert(cid_dict)

adj = list2graph(chem_CTodbGene_interaction_list, chem_idx_map)
adj = normalize_adj(adj)
adj = adj.tocoo()

adj = vec2tensor(adj)

with open('data/processed_data/chem_CTodbComm_graph.pkl','wb') as fw:
    pkl.dump(adj, fw)