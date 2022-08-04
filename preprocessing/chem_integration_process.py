# -*- coding: utf-8 -*-

import torch
import pickle as pkl
import scipy.sparse as sp
from Models.Utils import normalize_adj, vec2tensor, tensor2spcsr


# input multiscale graphs
with open('data/processed_data/chem_chem_networks.pkl', 'rb') as fb:
    ChemChem_Networks = pkl.load(fb)

with open('data/processed_data/chem_CTodbComm_graph.pkl', 'rb') as fb:
    ChemGene_CTodb = pkl.load(fb)

with open('data/processed_data/chem_DGidbComm_graph.pkl', 'rb') as fb:
    ChemGene_DGidb = pkl.load(fb)

with open('data/processed_data/chem_database_networks.pkl', 'rb') as fb:
    ChemString_Networks = pkl.load(fb)

with open('data/processed_data/chem_DisGeneComm_graph.pkl', 'rb') as fb:
    ChemDis_Networks = pkl.load(fb)

# integartion
ChemGene_DGidb_dense = ChemGene_DGidb.to_dense()
ChemGene_CTodb_dense = ChemGene_CTodb.to_dense()
ChemDis_Networks_dense = ChemDis_Networks.to_dense()

Combine_ChemGene_dense = ChemGene_DGidb_dense + ChemGene_CTodb_dense
Combine_ChemGeneDis_dense = Combine_ChemGene_dense + ChemDis_Networks_dense

Combine_ChemGene_scipy_csr = tensor2spcsr(Combine_ChemGene_dense)
Combine_ChemGeneDis_scipy_csr = tensor2spcsr(Combine_ChemGeneDis_dense)

Combine_ChemGene_scipy_csr = normalize_adj(Combine_ChemGene_scipy_csr)
Combine_ChemGeneDis_scipy_csr = normalize_adj(Combine_ChemGeneDis_scipy_csr)

Combine_ChemGene_networks = vec2tensor(Combine_ChemGene_scipy_csr.tocoo())
Combine_ChemGeneDis_networks = vec2tensor(Combine_ChemGeneDis_scipy_csr.tocoo())
with open('data/processed_data/Combine_ChemGene_networks.pkl','wb') as fw:
    pkl.dump(Combine_ChemGene_networks, fw)
with open('data/processed_data/Combine_ChemGeneDis_networks.pkl','wb') as fw:
    pkl.dump(Combine_ChemGeneDis_networks, fw)