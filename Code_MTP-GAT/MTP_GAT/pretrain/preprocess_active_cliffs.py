
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import torch_geometric.transforms as T
from MTP_GAT.utils.data import mol_to_graph_data_obj_super_rich, mol_to_graph_data_obj_rich
import pickle
from torch_geometric.data import Data 

def get_fp(mol, fp_type='ecfp4', fp_dim=1024):
    """
    Generates molecular fingerprints.
    Returns a NumPy array.
    """
    if mol is None:
        dim = fp_dim if fp_type != 'maccs' else 167 
        return np.zeros(dim, dtype=np.float32)

    if fp_type == 'ecfp4':
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
    elif fp_type == 'pubchem':
        fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=881) 
    elif fp_type == 'maccs':
        fp = AllChem.GetMACCSKeysFingerprint(mol)
    else:
        raise ValueError(f"Unsupported fingerprint type: {fp_type}. Choose from: 'ecfp4', 'pubchem', 'maccs'")

    arr = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)

def get_graph_data(mol, feat_type, transform, fp_dim):
    """Converts an RDKit Mol object to a PyG Data object."""
    if mol is None:
        dummy_node_feat_dim = 143 if feat_type == 'rich' else 143
        dummy_edge_feat_dim = 14
        data = Data(
            x=torch.zeros(1, dummy_node_feat_dim),
            edge_index=torch.empty((2,0), dtype=torch.long),
            edge_attr=torch.empty((0, dummy_edge_feat_dim))
        )
        data.pe = torch.zeros(1, transform.walk_length, dtype=torch.float)

        data.mol_fps = torch.zeros(1, fp_dim, dtype=torch.float)
        data.mol_md = torch.tensor(0.0)
        return data
    elif feat_type == 'super_rich':
        data = mol_to_graph_data_obj_super_rich(mol)
    elif feat_type == 'rich':
        data = mol_to_graph_data_obj_rich(mol)
    else:
        raise ValueError(f"Unknown feature type: {feat_type}")
    
    if hasattr(data, 'x') and data.x.size(0) > 0:
        data = transform(data)
    else:
        pe_size = data.x.size(0) if data.x.size(0) > 0 else 1
        data.pe = torch.zeros(pe_size, transform.walk_length, dtype=torch.float)
    
    return data

def find_additional_negatives(all_unique_smiles, all_mols, all_fps, smiles_to_plc50_map, anchor_smi, anchor_plc50, num_to_find, exclude_smis, sim_threshold, activity_diff_threshold, fp_dim):
    """
    Finds additional negative samples that are structurally dissimilar but have high activity difference.
    """
    additional_negatives = []
    anchor_mol = all_mols.get(anchor_smi)
    if anchor_mol is None:
        return [] 

    anchor_fp_np = all_fps.get(anchor_smi)
    if anchor_fp_np is None:
        return []
    
    anchor_bv = DataStructs.ExplicitBitVect(len(anchor_fp_np))
    for i, bit_val in enumerate(anchor_fp_np):
        if bit_val > 0:
            anchor_bv.SetBit(i)

    for candidate_smi in all_unique_smiles:
        if len(additional_negatives) >= num_to_find:
            break 

        if candidate_smi in exclude_smis:
            continue 

        candidate_mol = all_mols.get(candidate_smi)
        if candidate_mol is None:
            continue 

        candidate_fp_np = all_fps.get(candidate_smi)
        if candidate_fp_np is None:
            continue

        candidate_bv = DataStructs.ExplicitBitVect(len(candidate_fp_np))
        for i, bit_val in enumerate(candidate_fp_np):
            if bit_val > 0:
                candidate_bv.SetBit(i)

        similarity = DataStructs.TanimotoSimilarity(
            anchor_bv,
            candidate_bv
        )

        candidate_plc50 = smiles_to_plc50_map.get(candidate_smi)
        if candidate_plc50 is None:
            continue

        activity_diff = abs(anchor_plc50 - candidate_plc50)

        if similarity < sim_threshold and activity_diff > activity_diff_threshold:
            additional_negatives.append(candidate_smi)
            
    return additional_negatives[:num_to_find]

def preprocess_and_save(data_dir, data_file, output_dir, feat_type, fp_type, fp_dim, sim_threshold, activity_diff_threshold, num_additional_negatives):
    """
    Handles the entire preprocessing pipeline for a single data file.
    """
    output_file_name = f"{os.path.splitext(data_file)[0]}_{feat_type}_{fp_type}.pt"
    output_path = os.path.join(output_dir, output_file_name)
    
    if os.path.exists(output_path):
        print(f"✅ 预处理文件已存在：{output_path}。跳过。")
        return
        
    print(f"⏳ 正在预处理 {data_file}...")
    
    data_path = os.path.join(data_dir, data_file)
    df = pd.read_csv(data_path)
    
    all_smiles_list = (df['SMILES_anchor'].tolist() + df['SMILES_eng'].tolist() + df['SMILES_pos'].tolist())
    all_unique_smiles = list(set(all_smiles_list))
    
    all_mols = {smi: Chem.MolFromSmiles(smi) for smi in tqdm(all_unique_smiles, desc="解析所有SMILES")}
    all_mols = {smi: mol for smi, mol in all_mols.items() if mol is not None}
    
    def _get_fp_dim(fp_dim, fp_type):
        default_dims = {'ecfp4': 1024, 'pubchem': 881, 'maccs': 167}
        if fp_dim is not None: return fp_dim
        if fp_type not in default_dims: raise ValueError(f"不支持的指纹类型: {fp_type}")
        return default_dims[fp_type]
    
    actual_fp_dim = _get_fp_dim(fp_dim, fp_type)

    all_fps = {smi: get_fp(mol, fp_type, actual_fp_dim) for smi, mol in tqdm(all_mols.items(), desc="计算所有指纹")}
    
    smiles_to_plc50_map = {}
    for _, row in df.iterrows():
        if row['SMILES_anchor'] in all_mols: smiles_to_plc50_map[row['SMILES_anchor']] = row['pLC50_anchor']
        if row['SMILES_eng'] in all_mols: smiles_to_plc50_map[row['SMILES_eng']] = row['pLC50_eng']
        if row['SMILES_pos'] in all_mols: smiles_to_plc50_map[row['SMILES_pos']] = row['pLC50_pos']

    processed_data_list = []
    transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="生成数据条目"):
        anchor_smi = row['SMILES_anchor']
        pos_smi = row['SMILES_pos']
        explicit_neg_smi = row['SMILES_eng']
        
        if not (anchor_smi in all_mols and pos_smi in all_mols and explicit_neg_smi in all_mols):
            continue
            
        additional_neg_smis = find_additional_negatives(
            all_unique_smiles, all_mols, all_fps, smiles_to_plc50_map,
            anchor_smi, smiles_to_plc50_map[anchor_smi],
            num_additional_negatives,
            exclude_smis={anchor_smi, pos_smi, explicit_neg_smi},
            sim_threshold=sim_threshold, activity_diff_threshold=activity_diff_threshold, fp_dim=actual_fp_dim
        )
        
        triplet_smis = [anchor_smi, pos_smi, explicit_neg_smi] + additional_neg_smis
        
        data_objects = []
        for smi in triplet_smis:
            mol = all_mols.get(smi)
            data_obj = get_graph_data(mol, feat_type, transform, actual_fp_dim)
            
            if data_obj.x.size(0) == 0:
                 continue
            
            data_obj.smi = smi
            # ⚠️ 修改：将 mol_fp 改为 mol_fps
            data_obj.mol_fps = torch.from_numpy(all_fps[smi]).float()
            data_obj.mol_md = torch.tensor(smiles_to_plc50_map[smi]).float()
            data_objects.append(data_obj)

        if len(data_objects) >= 3:
            processed_data_list.append(data_objects)

    print(f"✅ 成功处理了 {len(processed_data_list)} 条数据。")
    
    torch.save(processed_data_list, output_path)
    print(f"💾 数据已保存到 {output_path}")

if __name__ == '__main__':
    BASE_DIR = ''
    DATA_DIR = os.path.join(BASE_DIR, 'data/pretrain')
    OUTPUT_CACHE_DIR = os.path.join(DATA_DIR, 'processed_cache')
    os.makedirs(OUTPUT_CACHE_DIR, exist_ok=True)

    CONFIG = {
        'feat_type': 'rich',
        'fp_type': 'ecfp4',
        'fp_dim': None,
        'sim_threshold': 0.5,
        'activity_diff_threshold': 2.0,
        'num_additional_negatives': 5
    }

    preprocess_and_save(
        DATA_DIR,
        OUTPUT_CACHE_DIR,
        **CONFIG
    )
