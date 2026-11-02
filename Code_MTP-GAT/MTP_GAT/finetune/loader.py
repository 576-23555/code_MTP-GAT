import os
import pickle
import pandas as pd
import scipy.sparse as sps
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import torch
from torch.utils.data import Dataset
import torch_geometric.transforms as T
from MTP_GAT.utils.data import mol_to_graph_data_obj_basic, mol_to_graph_data_obj_rich, mol_to_graph_data_obj_super_rich

class MoleculeDataset(Dataset):
    def __init__(self, data_dir, data_name, feat_type):
        self.feat_type = feat_type
        if data_name.endswith('.csv'):
            file_path = os.path.join(data_dir, data_name)
        elif data_name.endswith('.xlsx'):
            file_path = os.path.join(data_dir, data_name)
        else:
            file_path = os.path.join(data_dir, f'{data_name}.csv')

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件 '{file_path}' 不存在。")
            
        if file_path.endswith('.xlsx'):
            data = pd.read_excel(file_path)
        else: 
            data = pd.read_csv(file_path)
        
        smiles_raw = data['SMILES'].to_list()             
        labels_raw = data[['pIC50']].values 

        self.smiles = []
        self.labels = [] # Stores processed labels
        self.mol_data = [] # Stores PyTorch Geometric Data objects

        self.transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        
        for i, smi in enumerate(smiles_raw):
            mol = Chem.MolFromSmiles(smi)
            # Only process if Mol object is successfully created
            if mol is not None:
                try:
                    if self.feat_type == 'basic':
                        data_obj = mol_to_graph_data_obj_basic(mol)
                    elif self.feat_type == 'rich':
                        data_obj = mol_to_graph_data_obj_rich(mol)
                    elif self.feat_type == 'super_rich':
                        data_obj = mol_to_graph_data_obj_super_rich(mol)
                    else:
                        raise ValueError(f"不支持的 feat_type: {self.feat_type}")
                    
                    data_obj = self.transform(data_obj)
                    
                    self.smiles.append(smi)
                    self.labels.append(labels_raw[i])
                    self.mol_data.append(data_obj)
                except Exception as e:
                    # Handle cases where graph conversion might fail for some molecules
                    print(f"Warning: Could not process SMILES '{smi}' (index {i}). Error: {e}")
            else:
                print(f"Warning: Could not parse SMILES '{smi}' (index {i}) to molecule. Skipping.")

        if self.labels:
            self.num_task = self.labels[0].shape[0] if len(self.labels[0].shape) > 0 else 1
        else:
            self.num_task = 1
            print("Warning: No valid molecules were processed. num_task defaulted to 1.")


    def __len__(self):
        return len(self.mol_data)

    def __getitem__(self, idx):
        graph = self.mol_data[idx]
        
        graph.label = torch.Tensor(self.labels[idx])
        graph.smi = self.smiles[idx]
        
        return graph