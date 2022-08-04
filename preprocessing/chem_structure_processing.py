import numpy as np
import pickle as pkl
import scipy.sparse as sp
from Models.Utils import atomFeatures, bondFeatures, normalize_adj, read_smiles
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import pandas as pd
import torch
import argparse
import json
from networkx.readwrite import json_graph
from sklearn.model_selection import KFold

# chemical properties

FormalCharge = [-1, 1, 2, 3, 0]
ExplicitHs = [1, 2, 3, 0]
Degree = [1, 2, 3, 4, 5, 0]
Aromatic = [1]

InRing = [1]
self_loop = [1]

atom_list = ['B','C','N','O','F','P','I','S','Cr','Co','Cl','Br','Se','Hg']
bond_list = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

# processing

drug_smiles = {}

pd_csv = pd.read_csv('data/raw_data/DeepCE_drugs.csv')
for i in range(len(pd_csv['Smiles'])):
    """
    There are five redundant records in the DeepCE dataset:


    BRD-U86686840 VS BRD-K46056750
    BRD-U70626184 VS BRD-K64890080
    BRD-K86118762 VS BRD-K07667918
    BRD-K94012289 VS BRD-K87124298
    BRD-U07805514 VS BRD-K19540840

    if str(pd_csv['CID'][i]).strip() in drug_smiles:
        print('first: {}'.format(drug_smiles[str(pd_csv['CID'][i]).strip()]))
        print('second: {}'.format(pd_csv['BRD_ID'][i].strip()))
    drug_smiles[str(pd_csv['CID'][i]).strip()] = pd_csv['BRD_ID'][i].strip()
    """
    drug_smiles[str(pd_csv['CID'][i]).strip()] = pd_csv['Smiles'][i].strip()

ordered_drug_items = list(dict.fromkeys(drug_smiles.keys()))
ordered_drug_smiles = [drug_smiles[item] for item in ordered_drug_items]

smisuppl = np.array(ordered_drug_smiles)
print('count:', len(smisuppl))

n_max = 150
dim_atom = len(FormalCharge) - 1 + len(ExplicitHs) - 1 + len(Aromatic) + len(atom_list)
dim_bond = len(bond_list) + len(InRing) + len(self_loop)
dim_ConMap = dim_atom + (dim_bond - len(self_loop)) + dim_atom

current_max = 0

DA = np.empty(shape=(0, n_max, dim_atom), dtype=bool)
DB = np.empty(shape=(0, n_max, n_max, dim_bond), dtype=bool)
DC = np.empty(shape=(0, n_max, n_max, dim_ConMap), dtype=bool)
DY = []
Dsmi = []

da_list = []
db_list = []
ConMap_list = []

for i, smi in enumerate(smisuppl):

    if i % 100 == 0: print(i, len(Dsmi), current_max, flush=True)

    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False)
    mol = Chem.MolFromSmiles(smi)
    Chem.Kekulize(mol)

    n_atom = mol.GetNumHeavyAtoms()
    
    # constuct atom feature vector
    node_for_atom = np.zeros((n_atom, dim_atom), dtype=bool)
    for j in range(n_atom):
        atom = mol.GetAtomWithIdx(j)
        node_for_atom[j, :] = atomFeatures(atom, atom_list, FormalCharge, ExplicitHs, Degree)
        
    # constuct bond feature vector
    ConMap = np.zeros((n_atom, n_atom, dim_ConMap), dtype=bool)
    edge_for_bond = np.zeros((n_atom, n_atom, dim_bond), dtype=bool)
    for j in range(n_atom - 1):
        for k in range(j + 1, n_atom):
            bond = mol.GetBondBetweenAtoms(j, k)
            if bond is not None:
                atom_former = atomFeatures(mol.GetAtomWithIdx(j), atom_list, FormalCharge, ExplicitHs, Degree)
                bond_feat = bondFeatures(bond, bond_list)
                atom_latter = atomFeatures(mol.GetAtomWithIdx(k), atom_list, FormalCharge, ExplicitHs, Degree)
                ConMap[j, k, :] = np.concatenate([atom_former, bond_feat, atom_latter], axis=0)
                edge_for_bond[j, k, :(dim_bond-1)] = bondFeatures(bond, bond_list)
            
            # construct a symmetric vector
            edge_for_bond[k, j, :] = edge_for_bond[j, k, :]
            ConMap[k, j, :] = ConMap[j, k, :]

    for j in range(n_atom):
        edge_for_bond[j, j, -1] = 1

    # set current_max            
    if current_max < node_for_atom.shape[0]: current_max = node_for_atom.shape[0]
    
    node_for_atom = np.pad(node_for_atom, ((0, n_max - node_for_atom.shape[0]), (0, 0)))
    edge_for_bond = np.pad(edge_for_bond, ((0, n_max - edge_for_bond.shape[0]), (0, n_max - edge_for_bond.shape[1]), (0, 0)))
    ConMap = np.pad(ConMap, ((0, n_max - ConMap.shape[0]), (0, n_max - ConMap.shape[1]), (0, 0)))
    
    # property DY
    # candidates: GraphDescriptors.BalabanJ(mol), EState.EState.EStateIndices(mol)
    property_ = [Descriptors.ExactMolWt(mol), Descriptors.MolLogP(mol), Descriptors.MolMR(mol)]
    
    # append information
    da_list.append(node_for_atom)
    db_list.append(edge_for_bond)
    ConMap_list.append(ConMap)
    DY.append(property_)
#    Dsmi.append(smi)
    
    if i % 100 == 0 or i == len(smisuppl)-1:
        DA = np.concatenate([DA, np.array(da_list)], 0)
        DB = np.concatenate([DB, np.array(db_list)], 0)
        DC = np.concatenate([DC, np.array(ConMap_list)], 0)
        da_list = []
        db_list = []
        ConMap_list = []
        
DY = np.asarray(DY)

DA = DA[:,:current_max,:]
DB = DB[:,:current_max,:current_max,:]
DC = DC[:,:current_max,:current_max,:]

print(DA.shape, DB.shape, DC.shape, DY.shape)

DA = torch.FloatTensor(DA)
DB = torch.FloatTensor(DB)
DC = torch.FloatTensor(DC)
DY = torch.FloatTensor(DY)


# save
with open('data/processed_data/Smile_array.pkl','wb') as fw:
   pkl.dump([DA, DB, DC, DY], fw)