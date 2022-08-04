# -*- coding: utf-8 -*-

import yaml
import pickle as pkl
import scipy.sparse as sp
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from Models.Utils import idx_map_convert, list2graph, normalize_adj, vec2tensor, detection_association, remove_redundant

with open('data/raw_data/chem_web.pkl', 'rb') as fb:
    cid_dict = pkl.load(fb)

with open('data/processed_data/configurations.yml', 'r') as fr:
    params = yaml.load(fr, Loader=yaml.FullLoader)

with open('data/processed_data/gene_idx_map.pkl', 'rb') as fb:
    gene_idx_map = pkl.load(fb)
    
with open('data/raw_data/Gene2Comm_GreedyModularity.pkl', 'rb') as fb:
    Gene2Comm = pkl.load(fb)

with open('data/raw_data/all_gene_disease_associations.tsv', 'r') as f:

    Dis2Gene = {}

    head = f.readline()
    line = f.readline()
    while line:
        items = line.split('\t')
        GeneId = items[0].strip()
        DisId = items[4].strip()
        confidence_coeff = float(items[9])
        if confidence_coeff > 0.3: # customized
            if DisId not in Dis2Gene:
                Dis2Gene[DisId] = [GeneId]
            else:
                Dis2Gene[DisId].append(GeneId)
        line = f.readline()


with open('data/raw_data/disease_mappings.tsv', 'r') as f:

    MSHid2DisGeNETid = {}
    head = f.readline()
    line = f.readline()
    while line:
        items = line.split('\t')
        if items[2] == 'MSH' or items[2] == 'OMIM':
            DisGeNETid = items[0].strip()
            MSH_OMIM_id = items[3].strip()
            MSHid2DisGeNETid[MSH_OMIM_id] = DisGeNETid
        line = f.readline()

# incollected_dis = []
chem_dis_dict = {}
for key_1 in cid_dict.keys():
    chem_dis_dict[key_1] = []
    for key_2 in cid_dict[key_1]['CTD_dis']:
        if str(key_2) in MSHid2DisGeNETid:
            chem_dis_dict[key_1].append(MSHid2DisGeNETid[str(key_2)])
        # else:
        #     incollected_dis.append(key_1 + '\t' + key_2)


chem_DisGene_dict = {}

for key_1 in chem_dis_dict.keys():
    chem_DisGene_dict[key_1] = []
    for key_2 in chem_dis_dict[key_1]:
        if key_2 in Dis2Gene:
            chem_DisGene_dict[key_1].extend([gene_idx_map[k] for k in Dis2Gene[key_2] if k in gene_idx_map])

for k in chem_DisGene_dict.keys():

    chem_DisGene_dict[k] = remove_redundant(chem_DisGene_dict[k])


chem_DisGeneComm_dict = {}

for key_1 in chem_DisGene_dict.keys():
    chem_DisGeneComm_dict[key_1] = []
    for key_2 in chem_DisGene_dict[key_1]:
        chem_DisGeneComm_dict[key_1].append(Gene2Comm[key_2][0]) # as there are at most one element in each Gene2Comm item

chem_DisGene_interaction_list = []

for key_1 in chem_DisGeneComm_dict:
    for key_2 in chem_DisGeneComm_dict:
        if detection_association(chem_DisGeneComm_dict[key_1], chem_DisGeneComm_dict[key_2], params['threshold']):
            """
                threshold = 1: 600205 edges
                threshold = 2: 582216 edges
                threshold = 3: 566173 edges
            """
            chem_DisGene_interaction_list.append([key_1, key_2])

chem_idx_map = idx_map_convert(cid_dict)

adj = list2graph(chem_DisGene_interaction_list, chem_idx_map)
adj = normalize_adj(adj)
adj = adj.tocoo()

adj = vec2tensor(adj)

with open('data/processed_data/chem_DisGeneComm_graph.pkl','wb') as fw:
    pkl.dump(adj, fw)