from urllib import response
import urllib.request
import urllib.parse
import pickle as pkl
import json
import time

with open('/data/raw_data/DeepCE_drugs.csv', 'r') as r:
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


def opera_DGidb_chem_gene(response):
    gene_ids = []
    profiles = response.split('\n')
    for profile in profiles[1:-1]:
        gene_ids.append(profile.split(',')[0].strip())
    return gene_ids

def opera_CTD_chem_gene(response):
    gene_ids = []
    profiles = response.split('\n')
    for profile in profiles[1:-1]:
        gene_ids.append(profile.split(',')[4].strip())
    return gene_ids

def opera_DB_chem_chem(response):
    cid_ids = []
    profiles = response.split('\n')
    for profile in profiles[1:-1]:
        cid_ids.append(profile.split(',')[2].strip())
    return cid_ids

def opera_CTD_dis(response):
    disease_ids = []
    profiles = response.split('\n')
    for profile in profiles[1:-1]:
        # if '|' in profile.split(',')[4].strip():
        #     for disease_id in profile.split(',')[4].strip().split('|'):
        #         disease_ids.append(disease_id.strip())
        # else:
        disease_ids.append(profile.split(',')[4].strip())
    return disease_ids


cid_dict = {}

for line in lines:
    line = line.split(',')

    cid_id = line[3].strip()
    cid_dict[cid_id] = {}

    print(cid_id)

    # DGidb chem gene interaction
    request = urllib.request.Request(DGidb_chem_gene_1 + cid_id + DGidb_chem_gene_2)
    response = urllib.request.urlopen(request).read().decode('utf-8')

    cid_dict[cid_id]['DGidb_chem_gene'] = opera_DGidb_chem_gene(response)

    time.sleep(1)

    # CTD chem gene interaction
    request = urllib.request.Request(CTD_chem_gene_1 + cid_id + CTD_chem_gene_2)
    response = urllib.request.urlopen(request).read().decode('utf-8')

    cid_dict[cid_id]['CTD_chem_gene'] = opera_CTD_chem_gene(response)
    
    time.sleep(1)

    # DrugBank chem chem interaction
    request = urllib.request.Request(DB_chem_chem_1 + cid_id + DB_chem_chem_2)
    response = urllib.request.urlopen(request).read().decode('utf-8')

    cid_dict[cid_id]['DB_chem_chem'] = opera_DB_chem_chem(response)
    
    time.sleep(1)

    # CTD associated disorders and diseases
    request = urllib.request.Request(CTD_dis_1 + cid_id + CTD_dis_2)
    response = urllib.request.urlopen(request).read().decode('utf-8')

    cid_dict[cid_id]['CTD_dis'] = opera_CTD_dis(response)

    time.sleep(1)


with open('data/processed_data/chem_web.pkl','wb') as fw:
    pkl.dump(cid_dict, fw)