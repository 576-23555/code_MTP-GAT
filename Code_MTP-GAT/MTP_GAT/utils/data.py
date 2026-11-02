import copy
import json
import os
import torch
import pickle
import collections
from collections import Counter
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MakeScaffoldGeneric, MurckoScaffoldSmiles # type: ignore
from torch.utils import data
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset, Batch
from torch_geometric.utils import to_networkx
from torch_geometric.transforms.virtual_node import VirtualNode

hydrogen_donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]") 
hydrogen_acceptor = Chem.MolFromSmarts(
    "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),"
    "n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]")  
acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")  
basic = Chem.MolFromSmarts(
    "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);"
    "!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]") 

allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)), 
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], 
    'possible_chirality_list': [ 
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list': [ 
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8], 
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6], 
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  
    'possible_bonds': [  
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [  
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ],
    'possible_bond_stereo': [ 
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS
    ]
}

get_fp = lambda mol: AllChem.GetMorganFingerprintAsBitVect(
    mol, radius=2, nBits=512) 

def get_bond_feature_name(bond):
    """
    Returns a string representation of the RDKit bond type.
    """
    bt = bond.GetBondType()
    if bt == Chem.rdchem.BondType.SINGLE:
        return "SINGLE"
    elif bt == Chem.rdchem.BondType.DOUBLE:
        return "DOUBLE"
    elif bt == Chem.rdchem.BondType.TRIPLE:
        return "TRIPLE"
    elif bt == Chem.rdchem.BondType.AROMATIC:
        return "AROMATIC"
    # For any other bond types or if you want a more general string representation
    return str(bt)

def get_bond_feature_dims():
    """
    Returns the total dimension of bond features used in mol_to_graph_data_obj_super_rich.
    Based on the current implementation:
    BondType (5) + GetStereo (6) + GetIsConjugated (1) + IsInRing (1) = 13
    """
    return 13

def get_scaffold(smi, generic=False, return_mol=False):

    if generic:
        assert not return_mol

    if isinstance(smi, str):
        mol = Chem.MolFromSmiles(smi)
    else:
        mol = smi

    if generic:
        if return_mol:
            return MakeScaffoldGeneric(mol)  # 通用骨架分子
        else:
            return Chem.MolToSmiles(MakeScaffoldGeneric(mol))  # 通用骨架SMILES
    else:
        if return_mol:
            return GetScaffoldForMol(mol)  
            return Chem.MolToSmiles(GetScaffoldForMol(mol)) 

def canonicalize(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))

def onek_encoding_unk(value, choices):

    encoding = [0] * (len(choices) + 1)
    try:
        index = choices.index(value)
    except ValueError:
        index = len(choices)
    encoding[index] = 1
    return encoding

def mol_to_graph_data_obj_super_rich(mol, virtual_node=False):

    hydrogen_donor_match = sum(mol.GetSubstructMatches(hydrogen_donor), ())
    hydrogen_acceptor_match = sum(mol.GetSubstructMatches(hydrogen_acceptor), ())
    acidic_match = sum(mol.GetSubstructMatches(acidic), ())
    basic_match = sum(mol.GetSubstructMatches(basic), ())
    ring_info = mol.GetRingInfo()

    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        feature = onek_encoding_unk(atom.GetAtomicNum(), list(range(119))) + \
                  onek_encoding_unk(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5]) + \
                  onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0]) + \
                  onek_encoding_unk(atom.GetChiralTag(), [0, 1, 2, 3]) + \
                  onek_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) + \
                  onek_encoding_unk(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                                               Chem.rdchem.HybridizationType.SP2,
                                                               Chem.rdchem.HybridizationType.SP3,
                                                               Chem.rdchem.HybridizationType.SP3D,
                                                               Chem.rdchem.HybridizationType.SP3D2,
                                                               Chem.rdchem.HybridizationType.UNSPECIFIED 
                                                              ]) + \
                  [1 if atom.GetIsAromatic() else 0] + \
                  [atom.GetMass() * 0.01] + \
                  onek_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                  [atom_idx in hydrogen_acceptor_match] + \
                  [atom_idx in hydrogen_donor_match] + \
                  [atom_idx in acidic_match] + \
                  [atom_idx in basic_match] + \
                  [ring_info.IsAtomInRingOfSize(atom_idx, 3),
                   ring_info.IsAtomInRingOfSize(atom_idx, 4),
                   ring_info.IsAtomInRingOfSize(atom_idx, 5),
                   ring_info.IsAtomInRingOfSize(atom_idx, 6),
                   ring_info.IsAtomInRingOfSize(atom_idx, 7),
                   ring_info.IsAtomInRingOfSize(atom_idx, 8)]

        atom_features_list.append(feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    num_bond_features = 13 
    if len(mol.GetBonds()) > 0: 
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            feature = onek_encoding_unk(bond.GetBondType(), allowable_features['possible_bonds']) + \
                      onek_encoding_unk(int(bond.GetStereo()), list(range(6))) + \
                      [bond.GetIsConjugated()] + \
                      [bond.IsInRing()]

            edges_list.append((i, j))
            edge_features_list.append(feature)
            edges_list.append((j, i))  
            edge_features_list.append(feature)

        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else: 
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    if virtual_node: 
        data = VirtualNode()(data)

    return data

def mol_to_graph_data_obj_rich(mol, virtual_node=False):
    atom_features_list = []
    for atom in mol.GetAtoms():
        feature = onek_encoding_unk(atom.GetAtomicNum(), list(range(1, 101))) + \
                  onek_encoding_unk(atom.GetDegree(), list(range(11))) + \
                  onek_encoding_unk(atom.GetFormalCharge(), list(range(-2, 3))) + \
                  onek_encoding_unk(atom.GetNumRadicalElectrons(), list(range(5))) + \
                  onek_encoding_unk(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                                               Chem.rdchem.HybridizationType.SP2,
                                                               Chem.rdchem.HybridizationType.SP3,
                                                               Chem.rdchem.HybridizationType.SP3D,
                                                               Chem.rdchem.HybridizationType.SP3D2,
                                                               Chem.rdchem.HybridizationType.UNSPECIFIED
                                                              ]) + \
                  [1 if atom.GetIsAromatic() else 0] + \
                  onek_encoding_unk(atom.GetTotalNumHs(), list(range(5))) + \
                  [1 if atom.HasProp('_ChiralityPossible') else 0]
        if not atom.HasProp('_CIPCode'):
            feature += [0, 0, 0]
        else:
            feature += onek_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S']) 
        feature += [atom.GetMass() * 0.01]
        atom_features_list.append(feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    num_bond_features = 14 
    if len(mol.GetBonds()) > 0:
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            feature = onek_encoding_unk(bond.GetBondType(), [Chem.rdchem.BondType.SINGLE,
                                                              Chem.rdchem.BondType.DOUBLE,
                                                              Chem.rdchem.BondType.TRIPLE,
                                                              Chem.rdchem.BondType.AROMATIC]) + \
                      [1 if bond.GetIsConjugated() else 0] + \
                      [1 if bond.IsInRing() else 0] + \
                      onek_encoding_unk(bond.GetStereo(), [Chem.rdchem.BondStereo.STEREONONE,
                                                             Chem.rdchem.BondStereo.STEREOANY,
                                                             Chem.rdchem.BondStereo.STEREOZ,
                                                             Chem.rdchem.BondStereo.STEREOE,
                                                             Chem.rdchem.BondStereo.STEREOCIS,
                                                             Chem.rdchem.BondStereo.STEREOTRANS])
            edges_list.append((i, j))
            edge_features_list.append(feature)
            edges_list.append((j, i))
            edge_features_list.append(feature)

        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if virtual_node:
        data = VirtualNode()(data)
    return data

def mol_to_graph_data_obj_basic(mol, virtual_node=False):

    num_atom_features = 2  
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
                                         'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    num_bond_features = 2  
    if len(mol.GetBonds()) > 0:
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if virtual_node:
        data = VirtualNode()(data)
    return data

def graph_data_obj_to_mol_simple(data_x, data_edge_index, data_edge_attr):

    mol = Chem.RWMol()

    atom_features = data_x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        atomic_num_idx, chirality_tag_idx = atom_features[i]
        atomic_num = allowable_features['possible_atomic_num_list'][atomic_num_idx]
        chirality_tag = allowable_features['possible_chirality_list'][chirality_tag_idx]
        atom = Chem.Atom(atomic_num)
        atom.SetChiralTag(chirality_tag)
        mol.AddAtom(atom)

    edge_index = data_edge_index.cpu().numpy()
    edge_attr = data_edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        bond_type_idx, bond_dir_idx = edge_attr[j]
        bond_type = allowable_features['possible_bonds'][bond_type_idx]
        bond_dir = allowable_features['possible_bond_dirs'][bond_dir_idx]
        mol.AddBond(begin_idx, end_idx, bond_type)
        new_bond = mol.GetBondBetweenAtoms(begin_idx, end_idx)
        new_bond.SetBondDir(bond_dir)

    return mol

def list_collate_fn(data_list):

    data_list = [d for d in data_list if d is not None]
    if not data_list:
        return None 
    num_candidates = len(data_list[0])
    data_full = []
    for d in data_list:
        data_full.extend(d)
    batch = Batch.from_data_list(data_full)
    return batch

def list_collate_fn_pretrain(data_list):
    data_list = [d for d in data_list if d is not None]
    if not data_list: 
        return None, None
    num_candidates = len(data_list[0])
    data_full = []
    data_anchor = []
    for d in data_list:
        data_full.extend(d)
        data_anchor.append(d[0])
    batch = Batch.from_data_list(data_full)
    return batch, data_anchor

def atom_to_vocab(mol, atom):
    nei = Counter()
    for a in atom.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), a.GetIdx())
        nei[str(a.GetSymbol()) + "-" + str(bond.GetBondType())] += 1
    keys = nei.keys()
    keys = list(keys)
    keys.sort()
    output = atom.GetSymbol()
    for k in keys:
        output = "%s_%s%d" % (output, k, nei[k])
    return output

def bond_to_vocab(mol, bond):
    nei = Counter()
    two_neighbors = (bond.GetBeginAtom(), bond.GetEndAtom())
    two_indices = [a.GetIdx() for a in two_neighbors]
    for nei_atom in two_neighbors:
        for a in nei_atom.GetNeighbors():
            a_idx = a.GetIdx()
            if a_idx in two_indices:
                continue
            tmp_bond = mol.GetBondBetweenAtoms(nei_atom.GetIdx(), a_idx)
            nei[str(nei_atom.GetSymbol()) + '-' + get_bond_feature_name(tmp_bond)] += 1
    keys = list(nei.keys())
    keys.sort()
    output = get_bond_feature_name(bond)
    for k in keys:
        output = "%s_%s%d" % (output, k, nei[k])
    return output

from rdkit import Chem


def generate_motif_features(mol, verbose=False, return_name=True):

    smarts_list = [
        ('Carboxylic Acid', 'C(=O)[O;H1,-]'),
        ('Amine', '[N;!H0;v3,v4&+1]'),
        ('Alcohol', '[O;H1;v2;!$(*-*=[O,N,P,S])]'),
        ('Aldehyde', '[C;H1](=O)'),
        ('Ketone', '[C](=O)'),
        ('Ester', 'C(=O)O'),
        ('Amide', 'C(=O)N'),
        ('Nitrile', 'C#N'),
        ('Thiols', '[S;H1]'),
        ('Phenol', 'c1ccccc1O'),
        ('Halides', '[Cl,Br,F,I]'),
        ('Ether', 'C-O-C'),
        ('Alkene', 'C=C'),
        ('Alkyne', 'C#C'),
        ('Nitro', '[N+](=O)[O-]'),
        ('Sulfonic Acid', 'S(=O)(=O)O'),
        ('Azide', '[N-]#[N+]=[N-]'),
        ('Isocyanate', 'C(=O)N=C'),
        ('Peroxide', '[O-O]'),
        ('Tert-Butyl', 'C(C)(C)C'),
        ('Benzene', 'c1ccccc1'),
        ('Pyridine', 'c1ccncc1'),
        ('Cyclohexane', 'C1CCCCC1')
    ]
    
    matched_frags = []
    output_smarts = []

    for name, smarts in smarts_list:
        substructure = Chem.MolFromSmarts(smarts) 
        if substructure is None:
            if verbose:
                print(f"Invalid SMARTS pattern: {smarts}")
            continue  
        
        if mol.HasSubstructMatch(substructure):  
            matched_frags.append(name) 
            output_smarts.append((name, smarts))
            if verbose:
                print(f"Found {name} with SMARTS: {smarts}")

    if len(matched_frags) == 0:
        if verbose:
            print("No functional groups found.")
    else:
        if verbose:
            print(f"Matched functional groups: {', '.join(matched_frags)}")

    return matched_frags, output_smarts
