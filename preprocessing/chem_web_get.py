# -*- coding: utf-8 -*-

import time
import pandas as pd
import pickle as pkl
from Models.Utils import remove_redundant

with open('data/raw_data/DeepCE_drugs.csv', 'r') as r:
    line = r.readline()
    lines = r.readlines()


# DGidb chem gene interaction
DGidb_chem_gene_1 = 'https://pubchem.ncbi.nlm.nih.gov/sdq/sdqagent.cgi?infmt=json&outfmt=csv&query={%22download%22:%22*%22,%22collection%22:%22dgidb%22,%22where%22:{%22ands%22:[{%22cid%22:%22'

DGidb_chem_gene_2 = '%22}]},%22order%22:[%22relevancescore,desc%22],%22start%22:1,%22limit%22:10000000,%22downloadfilename%22:%22CID_chem_dgidb%22}'

# CTD chem gene interaction
CTD_chem_gene_1 = 'https://pubchem.ncbi.nlm.nih.gov/sdq/sdqagent.cgi?infmt=json&outfmt=csv&query={%22download%22:%22*%22,%22collection%22:%22ctdchemicalgene%22,%22where%22:{%22ands%22:[{%22cid%22:%22'

CTD_chem_gene_2 = '%22}]},%22order%22:[%22relevancescore,desc%22],%22start%22:1,%22limit%22:10000000,%22downloadfilename%22:%22CID_chem_ctdchemicalgene%22}'

# DrugBank chem chem interaction
DB_chem_chem_1 = 'https://pubchem.ncbi.nlm.nih.gov/sdq/sdqagent.cgi?infmt=json&outfmt=csv&query={%22download%22:%22*%22,%22collection%22:%22drugbankddi%22,%22where%22:{%22ands%22:[{%22cid%22:%22'

DB_chem_chem_2 = '%22}]},%22order%22:[%22relevancescore,desc%22],%22start%22:1,%22limit%22:10000000,%22downloadfilename%22:%22CID_chem_drugbankddi%22}'

# CTD associated disorders and diseases
CTD_dis_1 = 'https://pubchem.ncbi.nlm.nih.gov/sdq/sdqagent.cgi?infmt=json&outfmt=csv&query={%22download%22:%22*%22,%22collection%22:%22ctd_chemical_disease%22,%22where%22:{%22ands%22:[{%22cid%22:%22'

CTD_dis_2 = '%22}]},%22order%22:[%22relevancescore,desc%22],%22start%22:1,%22limit%22:10000000,%22downloadfilename%22:%22CID_chem_ctd_chemical_disease%22}'


cid_dict = {}

for line in lines:
    line = line.split(',')

    cid_id = line[3].strip()
    cid_dict[cid_id] = {}

    print(cid_id)

    # DGidb chem gene interaction
    url = DGidb_chem_gene_1 + cid_id + DGidb_chem_gene_2
    dataframe_1 = pd.read_csv(url)
    cid_dict[cid_id]['DGidb_chem_gene'] = remove_redundant(dataframe_1['geneid'].tolist())
    time.sleep(1)

    # CTD chem gene interaction
    url = CTD_chem_gene_1 + cid_id + CTD_chem_gene_2
    dataframe_2 = pd.read_csv(url)
    cid_dict[cid_id]['CTD_chem_gene'] = remove_redundant(dataframe_2['geneid'][dataframe_2.taxid==9606].tolist())
    time.sleep(1)

    # DrugBank chem chem interaction
    url = DB_chem_chem_1 + cid_id + DB_chem_chem_2
    dataframe_3 = pd.read_csv(url)
    cid_dict[cid_id]['DB_chem_chem'] = remove_redundant(dataframe_3['cid2'].tolist())
    time.sleep(1)

    # CTD associated disorders and diseases
    url = CTD_dis_1 + cid_id + CTD_dis_2
    dataframe_4 = pd.read_csv(url)
    cid_dict[cid_id]['CTD_dis'] = remove_redundant(dataframe_4['diseaseextid'].tolist())
    time.sleep(1)

with open('data/raw_data/chem_web.pkl','wb') as fw:
    pkl.dump(cid_dict, fw)