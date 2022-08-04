# -*- coding: utf-8 -*-

import numpy as np
import torch
import pandas as pd
from Models.Utils import choose_mean_example


class DataReader():

    def __init__(self, data_file_train, data_file_test, data_file_val, cuda):

        self.feat_lib = {"time": "24H", "pert_id": ['BRD-U41416256', 'BRD-U60236422'], "pert_type": ["trt_cp"],
        "cell_id": ['A375', 'HA1E', 'HELA', 'HT29', 'MCF7', 'PC3', 'YAPC'],
        "pert_idose": ["0.04 um", "0.12 um", "0.37 um", "1.11 um", "3.33 um", "10.0 um"]}

        self.CID2BRDid = {}
        self.BRK_idx_map = {}
        self.cuda = cuda

        pd_csv = pd.read_csv('data/raw_data/DeepCE_drugs.csv')
        for i in range(len(pd_csv['Smiles'])):
            
            self.CID2BRDid[str(pd_csv['CID'][i]).strip()] = pd_csv['BRD_ID'][i].strip()

        for i, CID in enumerate(self.CID2BRDid.keys()):
            self.BRK_idx_map[self.CID2BRDid[CID]] = i


        feature_train, label_train = self.read_data(data_file_train, self.feat_lib)
        feature_test, label_test = self.read_data(data_file_test, self.feat_lib)
        feature_val, label_val = self.read_data(data_file_val, self.feat_lib)

        self.dose_time_dict, self.cell_line_dict, dosage_dict = self.get_feat_classification(feature_train)

        self.train_category_features, self.train_numerical_features, self.train_labels = self.word2vec(feature_train, label_train, self.BRK_idx_map, self.dose_time_dict, self.cell_line_dict, self.cuda)

        self.test_category_features, self.test_numerical_features, self.test_labels = self.word2vec(feature_test, label_test, self.BRK_idx_map, self.dose_time_dict, self.cell_line_dict, self.cuda)

        self.val_category_features, self.val_numerical_features, self.val_labels = self.word2vec(feature_val, label_val, self.BRK_idx_map, self.dose_time_dict, self.cell_line_dict, self.cuda)

    def read_data(self, input_file, feat_lib):

        features = []
        labels = []
        data = dict()

        with open(input_file, 'r') as f:

            f.readline()

            for line in f:

                # REP.A002_A375_24H:K08, BRD-K60230970, trt_cp, A375, 20.0 um
                line = line.strip().split(',')
                assert len(line) == 983, "Wrong format"

                if feat_lib["time"] in line[0] and line[1] not in feat_lib['pert_id'] and line[2] in feat_lib["pert_type"] and line[3] in feat_lib['cell_id'] and line[4] in feat_lib["pert_idose"] and line[1] in self.BRK_idx_map.keys():

                    dose_time = line[0].split('H:')[0].split('_')[-1]
                    ft = ','.join([line[1]] + [dose_time] + line[3:5])
                    lb = [float(i) for i in line[5:]]

                    if ft in data.keys():
                        data[ft].append(lb)
                    else:
                        data[ft] = [lb]

        for ft, lb in sorted(data.items()): # determine the No.
            ft = ft.split(',')
            features.append(ft)
            if len(lb) == 1:
                labels.append(lb[0])
            else:
                lb = choose_mean_example(lb)
                labels.append(lb)
        return np.array(features), np.array(labels)

    def get_feat_classification(self, features):

        dose_time_set = sorted(list(set(features[:, 1])))
        cell_line_set = sorted(list(set(features[:, 2])))
        dosage_set = sorted(list(set(features[:, 3])))


        dose_time_dict = dict(zip(dose_time_set, list(range(len(dose_time_set)))))

        cell_line_dict = dict(zip(cell_line_set, list(range(len(cell_line_set)))))
            
        dosage_dict = dict(zip(dosage_set, list(range(len(dosage_set)))))
            

        return dose_time_dict, cell_line_dict, dosage_dict

    def word2vec(self, features, labels, BRK_idx_map, dose_time_dict, cell_line_dict, cuda):

        drug_category_features = []
        drug_numerical_features = []

        for i in range(len(features)):

            feature = features[i]

            drug_category_features.append([])

            drug_category_features[-1].append(BRK_idx_map[feature[0]])
            drug_category_features[-1].append(dose_time_dict[feature[1]])
            drug_category_features[-1].append(cell_line_dict[feature[2]])

            drug_numerical_features.append(float(feature[3].strip(' um')))


        drug_category_features = torch.tensor(drug_category_features)
        drug_numerical_features = torch.tensor(drug_numerical_features).unsqueeze(1)
        drug_labels = torch.FloatTensor(labels)

        if cuda:
            drug_category_features = drug_category_features.cuda()
            drug_numerical_features = drug_numerical_features.cuda()
            drug_labels = drug_labels.cuda()

        return drug_category_features, drug_numerical_features, drug_labels


    def get_batch_data(self, dataset, batch_size, shuffle):

        if dataset == 'train':
            category_features = self.train_category_features
            numerical_features = self.train_numerical_features
            labels = self.train_labels
        elif dataset == 'test':
            category_features = self.test_category_features
            numerical_features = self.test_numerical_features
            labels = self.test_labels
        elif dataset == 'val':
            category_features = self.val_category_features
            numerical_features = self.val_numerical_features
            labels = self.val_labels

        if shuffle:
            index = torch.randperm(len(category_features)).long()
            index = index.numpy()
        for start_idx in range(0, len(category_features), batch_size):
            if shuffle:
                if (start_idx + batch_size) < len(category_features):
                    excerpt = index[start_idx:start_idx + batch_size]
                else:
                    excerpt = index[start_idx:]
            else:
                if (start_idx + batch_size) < len(category_features):
                    excerpt = slice(start_idx, start_idx + batch_size)
                else:
                    excerpt = slice(start_idx, len(category_features))

            category_feats = category_features[excerpt]
            numerical_feats = numerical_features[excerpt]
            express_labels = labels[excerpt]
                
            yield category_feats, numerical_feats, express_labels