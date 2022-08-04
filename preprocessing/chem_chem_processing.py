# -*- coding: utf-8 -*-

import pickle as pkl
from Models.Utils import idx_map_convert, list2graph, normalize_adj, vec2tensor

with open('data/raw_data/chem_web.pkl', 'rb') as fb:
    cid_dict = pkl.load(fb)

chem_chem_interaction_list = []

for key_1 in cid_dict.keys():
    for key_2 in cid_dict[key_1]['DB_chem_chem']:
        if str(key_2) in cid_dict.keys():
            chem_chem_interaction_list.append([key_1, str(key_2)])


idx_map = idx_map_convert(cid_dict)

adj = list2graph(chem_chem_interaction_list, idx_map)
adj = normalize_adj(adj)
adj = adj.tocoo()

adj = vec2tensor(adj)

with open('data/processed_data/chem_chem_networks.pkl','wb') as fw:
    pkl.dump(adj, fw)