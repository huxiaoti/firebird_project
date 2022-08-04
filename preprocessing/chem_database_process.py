# -*- coding: utf-8 -*-

import pickle as pkl
from Models.Utils import idx_map_convert, list2graph, normalize_adj, vec2tensor

with open('data/raw_data/chem_web.pkl', 'rb') as fb:
    cid_dict = pkl.load(fb)

with open('data/raw_data/chemical_chemical.links.detailed.v5.0.tsv') as f:
    header = f.readline()
    lines = f.readlines()

idx_map = idx_map_convert(cid_dict)

chem_database_dict = {}

for line in lines:
    line = line.strip().split('\t')
    k = line[0].lstrip('CIDsm0')
    if k in idx_map:
        chem_database_dict[k] = []
        if line[1].lstrip('CIDsm0') in idx_map.keys():
            if int(line[-1]) > 150: # set 150 as the cutoff
                chem_database_dict[k].append(line[1].lstrip('CIDsm0'))
# removal redundancy
for key in chem_database_dict.keys():
    chem_database_dict[key] = list(set(chem_database_dict[key]))

chem_database_association_list = []

for key_1 in chem_database_dict.keys():
    for key_2 in chem_database_dict[key_1]:
        if key_2 in idx_map.keys():
            chem_database_association_list.append([key_1, key_2])

adj = list2graph(chem_database_association_list, idx_map)
adj = normalize_adj(adj)
adj = adj.tocoo()

adj = vec2tensor(adj)

with open('data/processed_data/chem_database_networks.pkl','wb') as fw:
    pkl.dump(adj, fw)