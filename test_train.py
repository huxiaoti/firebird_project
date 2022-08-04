# -*- coding: utf-8 -*-
"""
    @File    :   test_study.py
    @Time    :   2021/12/30 15:54:47
    @Author  :   Hu Xiaotian (Wu Shiauthie) 
    @Contact :   wuuwst@zju.edu.cn
"""

import os
import time
import random
import argparse
import yaml
import numpy as np
import pickle as pkl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from random import sample

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from Models.Utils import calc_roc, cross_val, load_data, clip_gradient, set_random_seed
from Models.Losses import MSELoss
from Models.Optims import ScheduledOptim, PolyLr
from Models import FireBird
from Models.DataGenerator import DataReader

from Models.Metric import rmse, correlation, precision_k

# Parameter settings
print('-- load parameters')

with open('data/processed_data/configurations.yml', 'r') as fr:
    params = yaml.load(fr, Loader=yaml.FullLoader)

# GPU settings
print('-- set GPUs')

params['cuda'] = params['cuda'] and torch.cuda.is_available()

if params['cuda']:
    torch.cuda.set_device('cuda:0')

# Random seed setting

set_random_seed(params['seed'], params['cuda'])

# Data Library Loading
print('-- load library')
adj_multi_ddis = load_data(params['multi_graphs'])

with open('data/processed_data/Gene_Asso.pkl', 'rb') as fb:
    Express_gene_adj = pkl.load(fb)
Express_gene_adj = Express_gene_adj.float()

with open('data/processed_data/Weighted_Gene_Asso.pkl', 'rb') as fb:
    Weight_gene_relations = pkl.load(fb)
Weight_gene_relations = Weight_gene_relations.float()

with open('data/processed_data/Smile_array.pkl', 'rb') as fb:
    [DA, DB, DC, DY] = pkl.load(fb)

DC = F.pad(DC, (0, 0, 0, params['img_size_Swin'] - DC.shape[2], 0, params['img_size_Swin'] - DC.shape[1]), 'constant', 0)
DC = DC.permute(0, 3, 1, 2)

# Data Indices Loading
print('-- load indices')
# Drug, Time, Cell line
index_data = DataReader('data/processed_data/signature_train.csv','data/processed_data/signature_test.csv','data/processed_data/signature_val.csv',params['cuda'])

cell_id =  index_data.cell_line_dict
dosing_time_lib = index_data.dose_time_dict

'''
num_batch = 256
drug_index = torch.randint(0, 1000, size=(num_batch, 1))
time_index = torch.randint(0, 3, size=(num_batch, 1))
cell_line_index = torch.randint(0, 3, size=(num_batch, 1))
# data_indices: [[2, 2, 0],[21, 0, 1],[1, 2, 3]]
# dose_indices: [[2.3],[5],[1.5]]
data_indices = torch.cat([drug_index,time_index,cell_line_index], dim=-1)
dose_indices = torch.rand(size=(num_batch,1))
dosing_time_lib = {'0':0, '3':1, '6':2, '12':3, '24':4}
dosing_time_indices = ['24'] * num_batch
dosing_time_indices = [dosing_time_lib[k] for k in dosing_time_indices]
'''

# Model building
print('-- construct model')

model = FireBird(alpha=params['alpha'],
                 dim_DA = DA.shape[-1],
                 dim_DB = DB.shape[-1],
                 dim_mol_props = DY.shape[-1],
                 num_landmarks= Express_gene_adj.shape[0],
                 dim_hid_GCN = params['dim_hid_GCN'],
                 img_size_Swin = params['img_size_Swin'],
                 patch_size_Swin = params['patch_size_Swin'],
                 dim_inchannel_Swin = DC.shape[1],
                 window_size_Swin = params['window_size_Swin'],
                 dim_outchannel_Swin = params['dim_outchannel_Swin'],
                 depths_Swin = params['depths_Swin'],
                 num_head_Swin = params['num_head_Swin'],
                 dim_out_Swin = params['dim_out_Swin'],
                 dim_hid_RGCN = params['dim_hid_RGCN'],
                 num_bases_RGCN = params['num_bases_RGCN'],
                 num_layer_RGCN = params['num_layer_RGCN'],
                 dim_out_RGCN = params['dim_out_RGCN'],
                 dim_hid_GAT = params['dim_hid_GAT'],
                 dim_out_GAT = params['dim_out_GAT'],
                 num_head_GAT = params['num_head_GAT'],
                 multi_graphs = params['multi_graphs'],
                 num_cell_lines = len(cell_id),
                 num_layers_Trans = params['num_layer_Trans'],
                 num_heads_Trans = params['num_head_Trans'],
                 dim_head_Trans = params['dim_head_Trans'],
                 dosing_time_lib = dosing_time_lib,
                 dropout = params['dropout'],
                 cuda = params['cuda'])

optimizer = optim.AdamW(model.parameters(), lr=params['lr'])
scheduler = PolyLr(optimizer, max_iteration=params['max_iteration'], warmup_iteration=params['warmup_iteration'])

if params['cuda'] == True:

    model.cuda()
    DA = DA.cuda()
    DB = DB.cuda()
    DC = DC.cuda()
    DY = DY.cuda()
    Express_gene_adj = Express_gene_adj.cuda()
    Weight_gene_relations = Weight_gene_relations.cuda()
    for key in adj_multi_ddis:
        adj_multi_ddis[key] = adj_multi_ddis[key].to_dense().cuda()
    

# Model training
print('-- fit model')

rmse_list_val = []
pearson_list_val = []
spearman_list_val = []
precisionk_list_val = []
precision_degree = [10, 20, 50, 100]

for n in range(params['max_iteration']):

    print("\nIteration %d:" % (n+1))

    model.train()
    epoch_loss = 0

    for i, batch in enumerate(index_data.get_batch_data('train', params['batch_size'], True)):

        category_feats, numerical_feats, express_labels = batch

        optimizer.zero_grad()

        outputs = model(DB, DA, DY, DC, params['multi_graphs'], adj_multi_ddis, Express_gene_adj, Weight_gene_relations, category_feats, numerical_feats)

        loss_train = MSELoss(outputs, express_labels)
        loss_train.backward()
        optimizer.step()
        # scheduler.step()

        epoch_loss += loss_train.item()

    print('Train loss: {:.4f}'.format(epoch_loss/(i+1)))

    model.eval()
    epoch_loss = 0

    label_np = np.empty([0, Express_gene_adj.shape[0]])
    predict_np = np.empty([0, Express_gene_adj.shape[0]])

    for i, batch in enumerate(index_data.get_batch_data('val', params['batch_size'], False)):

        category_feats, numerical_feats, express_labels = batch

        outputs = model(DB, DA, DY, DC, params['multi_graphs'], adj_multi_ddis, Express_gene_adj, Weight_gene_relations, category_feats, numerical_feats)

        loss_val = MSELoss(outputs, express_labels)
        epoch_loss += loss_val.item()
        label_np = np.concatenate((label_np, express_labels.cpu().detach().numpy()), axis=0)
        predict_np = np.concatenate((predict_np, outputs.cpu().detach().numpy()), axis=0)

    print('Val loss: {:.4f}'.format(epoch_loss / (i + 1)))

    rmse_score = rmse(label_np, predict_np)
    rmse_list_val.append(rmse_score)
    print('RMSE: %.4f' % rmse_score)
    pearson, _ = correlation(label_np, predict_np, 'pearson')
    pearson_list_val.append(pearson)
    print('Pearson\'s correlation: %.4f' % pearson)
    spearman, _ = correlation(label_np, predict_np, 'spearman')
    spearman_list_val.append(spearman)
    print('Spearman\'s correlation: %.4f' % spearman)
    precision = []
    for k in precision_degree:
        precision_neg, precision_pos = precision_k(label_np, predict_np, k)
        print("Precision@%d Positive: %.4f" % (k, precision_pos))
        print("Precision@%d Negative: %.4f" % (k, precision_neg))
        precision.append([precision_pos, precision_neg])
    precisionk_list_val.append(precision)