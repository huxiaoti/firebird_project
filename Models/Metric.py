# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

def rmse(labels, predicts):
    return np.sqrt(mean_squared_error(labels, predicts))


def correlation(labels, predicts, correlation_type):
    if correlation_type == 'pearson':
        corr = pearsonr
    elif correlation_type == 'spearman':
        corr = spearmanr
    else:
        raise ValueError("Unknown correlation type: %s" % correlation_type)
    score = []
    for label, predict in zip(labels, predicts):
        score.append(corr(label, predict)[0])
    return np.mean(score), score


def precision_k(labels, predicts, k):
    num_pos = 100
    num_neg = 100
    labels = np.argsort(labels, axis=1)
    predicts = np.argsort(predicts, axis=1)
    precision_k_neg = []
    precision_k_pos = []
    neg_label_set = labels[:, :num_neg]
    pos_label_set = labels[:, -num_pos:]
    neg_predict_set = predicts[:, :k]
    pos_predict_set = predicts[:, -k:]
    for i in range(len(neg_label_set)):
        neg_test = set(neg_label_set[i])
        pos_test = set(pos_label_set[i])
        neg_predict = set(neg_predict_set[i])
        pos_predict = set(pos_predict_set[i])
        precision_k_neg.append(len(neg_test.intersection(neg_predict)) / k)
        precision_k_pos.append(len(pos_test.intersection(pos_predict)) / k)
    return np.mean(precision_k_neg), np.mean(precision_k_pos)

 

# Cancer drug response
# presents half-maximal inhibitory concentration (IC50)
# The IC50 depicts the amount of drug needed to inhibit cancer cell growth by half. A smaller IC50 indicates that the drug is relatively more powerful
# Cited from
# DualGCN: a dual graph convolutional network model to predict cancer drug response

# 使用多个药物作用基因表达时间点，推测药物作用基因因果流，分析药物代谢动力学