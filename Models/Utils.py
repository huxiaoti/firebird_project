# -*- coding: utf-8 -*-


import numpy as np
import torch
import random
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.utils import validation
from rdkit import Chem
import scipy.sparse as sp
import pickle as pkl

import json
from networkx.readwrite import json_graph

def calc_roc(output_log, labels, test_idx):
    
    y_pred = torch.exp(output_log)[test_idx].cpu().detach().numpy()
    y_true = labels.squeeze().cpu().detach().numpy()[test_idx]
        
    acc = metrics.accuracy_score(y_true,  y_pred.argmax(axis=1))
    roc = metrics.roc_auc_score(y_true,  y_pred[:,1])
    precision, recall, _ = metrics.precision_recall_curve(y_true,  y_pred[:,1])
    aupr = metrics.auc(recall, precision)

    return acc, roc, aupr

def cross_val(train_rate, seed, drug):

    fold_time = int(1/(1-train_rate))
    kf = KFold(n_splits=fold_time, shuffle=True, random_state=seed)

    train_index_set = []
    test_index_set = []

    for train_index, test_index in kf.split(drug):
        train_index_set.append(train_index)
        test_index_set.append(test_index)

    return train_index_set, test_index_set


def set_random_seed(seed, cuda):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def load_data(multi_graph):
    """Load reference relation networks"""

    file2name = {
    'Chem_DGidbGene': 'chem_DGidbComm_graph',
    'Chem_DGodbGene': 'chem_CTodbComm_graph',
    'Chem_DisGene': 'chem_DisGeneComm_graph',
    'Chem_CombGene': 'Combine_ChemGene_networks',
    'Chem_CombDisGene': 'Combine_ChemGeneDis_networks',
    'Chem_Chem': 'chem_chem_networks',
    'Chem_String': 'chem_database_networks',
    }

    adj_multi_ddis = {}
    for graph in multi_graph:
        with open('data/processed_data/{}.pkl'.format(file2name[graph]),'rb') as f:
            adj = pkl.load(f)
        adj_multi_ddis[graph] = adj
        
    return adj_multi_ddis

def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)) # shape: (2708, 1433)
    r_inv_sqrt = np.power(rowsum, -0.5).flatten() # shape: (2708,)
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt) # shape: (2708, 2708)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def read_smiles(smiles_file):
    with open(smiles_file, 'r') as f:
        smiles_list = [line.strip() for line in f.readlines()]
    return smiles_list

def to_onehot(val, cat):

    vec = np.zeros(len(cat))
    for i, c in enumerate(cat):
        if val == c: vec[i] = 1

    if np.sum(vec) == 0: print('* exception: missing category: ', val, cat)
    assert np.sum(vec) == 1

    return vec


def atomFeatures(atom, atom_list, FormalCharge, ExplicitHs, Degree):

    v1 = to_onehot(atom.GetFormalCharge(), FormalCharge)[:-1]
    v2 = to_onehot(atom.GetNumExplicitHs(), ExplicitHs)[:-1]    
    v3 = to_onehot(atom.GetSymbol(), atom_list)
    # v4 = to_onehot(atom.GetDegree(), Degree)[:-1]
    # v5 = to_onehot(atom.GetImplicitValence(), [1, 2, 3, 0])[:-1]
    v4 = [1] if atom.GetIsAromatic() else [0]
    
    return np.concatenate([v1, v2, v3, v4], axis=0)

def bondFeatures(bond, bond_list):

    e1 = to_onehot(bond.GetBondType(), bond_list)
    ea = [1] if bond.IsInRing() else [0]
    e1 = np.concatenate([e1, ea], axis=0)

    return e1

def idx_map_convert(dict_):

    idx_map = {j: i for i, j in enumerate(dict_)}

    return idx_map


def list2graph(list_, idx_map):

    egde_array = np.array(list_)

    edges = np.array(list(map(idx_map.get, egde_array.flatten())), dtype=np.int32).reshape(egde_array.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(len(idx_map), len(idx_map)), dtype=np.float32)

    adj = adj + adj.T
    adj = adj + sp.eye(adj.shape[0])
    adj[adj > 1] = 1

    return adj

def tensor2spcsr(dense_tensor):

    dense_adj = torch.where(dense_tensor > 0, 1., 0.)
    adj = sp.csr_matrix(dense_adj.numpy())
    adj = adj + sp.eye(adj.shape[0])
    adj[adj > 0] = 1

    return adj

def normalize_adj(adj):

    """Row-normalize sparse matrix"""

    rowsum = np.array(adj.sum(1)) # shape: (2708,)
    r_inv_sqrt = np.power(rowsum, -0.5).flatten() # shape: (2708,)
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt) # shape: (2708, 2708)
    return adj.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt) # equal to first mlyiply by row, and then multiply by column

def vec2tensor(vector):

    values = vector.data
    indices = np.vstack((vector.row, vector.col))
    shape = vector.shape
    tensor = torch.sparse_coo_tensor(indices, values, shape)

    return tensor

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def remove_redundant(list_):

    new_list = list({}.fromkeys(list_).keys()) # list(dict.fromkeys(list_))

    return new_list

def detection_association(list_1, list_2, threshold):

    n = 0
    list_2_copy = list_2.copy() # list_2.deepcopy()
    for key_1 in list_1:
        if key_1 in list_2_copy:
            n += 1
            list_2_copy.remove(key_1)
            if n >= threshold:
                return True
    return False

def choose_mean_example(examples):

    num_example = len(examples)
    mean_value = (num_example - 1) / 2
    indexes = np.argsort(examples, axis=0)
    indexes = np.argsort(indexes, axis=0)
    indexes = np.mean(indexes, axis=1)
    distance = (indexes - mean_value)**2
    index = np.argmin(distance)
    
    return examples[index]