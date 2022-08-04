# -*- coding: utf-8 -*-

import numpy as np
import pickle as pkl
import scipy.sparse as sp
import networkx as nx
from Models.Utils import idx_map_convert

GeneName_list = []

with open('../../data/raw_data/DeepCE_gene_vector.csv', 'r') as f:
    
    line = f.readline()
    while line:
        GeneName = line.split(',')[0].strip()
        GeneName_list.append(GeneName)
        print(GeneName)
        line = f.readline()

# search in uniprot
print(','.join(GeneName_list))

GeneName2GeneID = {}

# info from uniprot retrieve
with open('../../data/raw_data/uniprot_yourlist_1018.tab', 'r') as f:

    line = f.readline()
    line = f.readline()
    while line:

        info = line.split('\t')
        GeneName = info[0]
        GeneID = info[6].strip().strip(';')

        if GeneName in GeneName2GeneID:
            print(GeneName)
            print(GeneID)
            print(GeneName2GeneID[GeneName])
            print('---')
        GeneName2GeneID[GeneName] = GeneID

        line = f.readline()

# artificial check for redundant records
GeneName2GeneID['HADH'] = '3033'
GeneName2GeneID['SCP2'] = '6342'
GeneName2GeneID['TRAP1'] = '10131'
GeneName2GeneID['LRP10'] = '26020'
GeneName2GeneID['CAT'] = '847'
GeneName2GeneID['ZFP36'] = '7538'
GeneName2GeneID['PNP'] = '4860'
GeneName2GeneID['NET1'] = '10276'
GeneName2GeneID['PAF1'] = '54623'
GeneName2GeneID['PCM1'] = '5108'
GeneName2GeneID['LIG1'] = '3978'
GeneName2GeneID['PRKACA'] = '5566'
GeneName2GeneID['POP4'] = '10775'
GeneName2GeneID['POLD4'] = '57804'
GeneName2GeneID['PAN2'] = '9924'
GeneName2GeneID['CDH3'] = '1001'
GeneName2GeneID['HES1'] = '3280'
GeneName2GeneID['FAS'] = '355'
GeneName2GeneID['PPOX'] = '5498'
GeneName2GeneID['ORC1'] = '4998'
GeneName2GeneID['NOS3'] = '4846'
GeneName2GeneID['RNH1'] = '6050'
GeneName2GeneID['CASK'] = '8573'
GeneName2GeneID['CHP1'] = '11261'
GeneName2GeneID['VAT1'] = '10493'
GeneName2GeneID['CAST'] = '831'
GeneName2GeneID['SFN'] = '2810'
GeneName2GeneID['ACAT2'] = '8435'
GeneName2GeneID['PAK1'] = '5058'
GeneName2GeneID['SPP1'] = '6696'
GeneName2GeneID['HN1L'] = '90861'
GeneName2GeneID['MIF'] = '4282'

for key in GeneName2GeneID.keys():
    if ';' in GeneName2GeneID[key]:
        print(key)

GeneName2GeneID['CALM3'] = '808'
GeneName2GeneID['HSPA1A'] = '3303'
GeneName2GeneID['CSNK1E'] = '1454'
GeneName2GeneID['PAK6'] = '56924'

with open('../../data/raw_data/LandmarkName2ID_dict.pkl','wb') as fw:
   pkl.dump(GeneName2GeneID, fw)
