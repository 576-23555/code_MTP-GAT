# preprocess_molecules.py
import os
import numpy as np
import torch
import pandas as pd
import pickle
from tqdm import tqdm
from rdkit import Chem
from rdkit import DataStructs
from MTP_GAT.utils.data import (
    mol_to_graph_data_obj_rich,
    mol_to_graph_data_obj_super_rich,
    onek_encoding_unk,
    atom_to_vocab,
    allowable_features,
    get_scaffold
)

def get_fp(mol, fp_type='ecfp4', fp_dim=1024):
    
    from rdkit.Chem import AllChem
    if mol is None:
        if fp_type == 'maccs':
            return np.zeros(167, dtype=np.float32)
        elif fp_type == 'pubchem':
            return np.zeros(881, dtype=np.float32)
        else:
            return np.zeros(fp_dim, dtype=np.float32)

    if fp_type == 'ecfp4':
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
    elif fp_type == 'pubchem':
        fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=881)
    elif fp_type == 'maccs':
        fp = AllChem.GetMACCSKeysFingerprint(mol)
    else:
        raise ValueError(f"不支持的指纹类型: {fp_type}. 可选: 'ecfp4', 'pubchem', 'maccs'")

    arr = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)

def subgraph_mask(mol, percent=0.15):

    node_num = mol.GetNumAtoms()
    mask_inds = set()
    if node_num == 0:
        return set()

    while len(mask_inds) < int(node_num * percent):
        available_indices = list(set(range(node_num)) - mask_inds)
        if not available_indices:
            break
        
        center_idx = np.random.choice(available_indices)
        queue = [center_idx]
        temp_mask_batch = {center_idx}
        
        while queue and len(mask_inds) + len(temp_mask_batch) <= int(node_num * percent):
            idx = queue.pop(0)
            atom = mol.GetAtomWithIdx(int(idx))
            for n_atom in atom.GetNeighbors():
                n_idx = n_atom.GetIdx()
                if n_idx not in mask_inds and n_idx not in temp_mask_batch:
                    temp_mask_batch.add(n_idx)
                    queue.append(n_idx)
        
        mask_inds.update(temp_mask_batch)
    return mask_inds

def get_graph_data_and_mask_inds(smi, feat_type='rich'):

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None
    
    try:
        data = eval(f'mol_to_graph_data_obj_{feat_type}')(mol)
    except Exception as e:
        print(f"Error converting SMILES {smi} to graph data: {e}")
        return None, None
    
    mask_inds = subgraph_mask(mol, percent=0.15)
    
    return data, mask_inds

def preprocess_and_save(data_dir, data_file, cache_dir, feat_type='rich', fp_type='ecfp4', fp_dim=None):
    """主函数，用于预处理并保存数据"""
    data_path = os.path.join(data_dir, data_file)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print(f"⏳ 正在预处理 {data_file}...")
    df = pd.read_csv(data_path)
    smiles_list = df['smiles'].tolist()
    
    data_list = []
    failed_smiles = 0

    for smi in tqdm(smiles_list, desc="Processing molecules"):
        data, mask_inds = get_graph_data_and_mask_inds(smi, feat_type=feat_type)
        
        if data is None:
            failed_smiles += 1
            continue
        mol = Chem.MolFromSmiles(smi)
        mol_fp = get_fp(mol, fp_type=fp_type, fp_dim=fp_dim)
        scaff_smi = get_scaffold(mol)
        scaff_mol = Chem.MolFromSmiles(scaff_smi)
        scaff_fp = get_fp(scaff_mol, fp_type=fp_type, fp_dim=fp_dim) if scaff_mol else mol_fp

        data.smi = smi
        data.mol_fp = torch.tensor(mol_fp, dtype=torch.float)
        data.scaff_fp = torch.tensor(scaff_fp, dtype=torch.float)
        data.mask_inds = list(mask_inds) 
        
        data_list.append(data)
        
    print(f"✅ 预处理完成。成功处理 {len(data_list)} 个分子，失败 {failed_smiles} 个。")
    
    os.makedirs(cache_dir, exist_ok=True)
    file_name = f"{os.path.splitext(data_file)[0]}_{feat_type}_{fp_type}.pt"
    save_path = os.path.join(cache_dir, file_name)
    torch.save(data_list, save_path)
    print(f"💾 数据已保存到 {save_path}")

if __name__ == '__main__':
    BASE_DIR = ''
    DATA_DIR = os.path.join(BASE_DIR, 'data/pretrain')
    OUTPUT_CACHE_DIR = os.path.join(DATA_DIR, 'processed_cache')
    os.makedirs(OUTPUT_CACHE_DIR, exist_ok=True)

    #配置预处理参数
    CONFIG = {
        'feat_type': 'rich', 
        'fp_type': 'ecfp4',
        'fp_dim': 1024,
    }
    preprocess_and_save(DATA_DIR, 'pretain.csv', OUTPUT_CACHE_DIR, **CONFIG)
 

