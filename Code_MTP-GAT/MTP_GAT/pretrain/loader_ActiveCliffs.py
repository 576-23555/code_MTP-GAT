
import os
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch_geometric.transforms as T
import pickle


def get_fp(mol, fp_type='ecfp4', fp_dim=1024):
    if mol is None:
        default_dims = {'ecfp4': 1024, 'pubchem': 881, 'maccs': 167}
        dim = default_dims.get(
            fp_type, fp_dim if fp_dim is not None else 1024
        )
        return np.zeros(dim, dtype=np.float32)

    if fp_type == 'ecfp4':
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
    elif fp_type == 'pubchem':
        fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=881)
    elif fp_type == 'maccs':
        fp = AllChem.GetMACCSKeysFingerprint(mol)
    else:
        raise ValueError(
            f"Unsupported fingerprint type: {fp_type}. "
            "Choose from: 'ecfp4', 'pubchem', 'maccs'"
        )

    arr = np.zeros((fp.GetNumBits(),), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)


class ActiveCliffsDataset(Dataset):
    def __init__(
        self,
        data_dir,
        data_file='pre_LC50.csv',
        feat_type='rich',
        fp_type='ecfp4',
        fp_dim=1024
    ):
        self.data_dir = data_dir
        self.data_file = data_file
        self.feat_type = feat_type
        self.fp_type = fp_type

        def _get_fp_dim(fp_dim, fp_type):
            default_dims = {
                'ecfp4': 1024,
                'pubchem': 881,
                'maccs': 167
            }
            if fp_dim is not None:
                return fp_dim
            if fp_type not in default_dims:
                raise ValueError(f"Unsupported fingerprint type: {fp_type}")
            return default_dims[fp_type]

        self.fp_dim = _get_fp_dim(fp_dim, fp_type)

        atom_vocab_path = os.path.join(data_dir, 'atom_vocab_pretrain.pkl')
        if not os.path.exists(atom_vocab_path):
            raise FileNotFoundError(
                f"Atom vocabulary file not found: {atom_vocab_path}"
            )

        print(f"Loading atom vocabulary: {atom_vocab_path}")
        with open(atom_vocab_path, 'rb') as f:
            self.atom_vocab_itos = pickle.load(f)
        print(
            f"Atom vocabulary loaded successfully. "
            f"Size: {len(self.atom_vocab_itos)}"
        )

        processed_file_name = (
            f"{os.path.splitext(data_file)[0]}_{feat_type}_{fp_type}.pt"
        )
        processed_path = os.path.join(
            data_dir, 'processed_cache', processed_file_name
        )

        if not os.path.exists(processed_path):
            raise FileNotFoundError(
                f"Preprocessed file not found: {processed_path}.\n"
                f"Please run the preprocessing script first with:\n"
                f"{{'feat_type': '{feat_type}', "
                f"'fp_type': '{fp_type}', 'fp_dim': {self.fp_dim}}}"
            )

        print(f"Loading preprocessed data: {processed_path}")
        self.data_entries = torch.load(processed_path)
        self.keys = list(range(len(self.data_entries)))
        print(f"Successfully loaded {len(self.keys)} samples.")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        data_objects = self.data_entries[idx]

        anchor_data = data_objects[0]
        pos_data = data_objects[1]
        neg_datas = data_objects[2:]

        return anchor_data, pos_data, neg_datas


if __name__ == '__main__':
    BASE_DIR = ''
    DATA_DIR = os.path.join(BASE_DIR, 'data/pretrain')

    CSV_FILE = 'pre_LC50.csv'

    print("Initializing dataset...")
    try:
        dataset = ActiveCliffsDataset(
            data_dir=DATA_DIR,
            data_file=CSV_FILE,
            feat_type='rich',
            fp_type='ecfp4',
            fp_dim=1024
        )

        print(
            f"Dataset size: {len(dataset)} samples | "
            f"Fingerprint type: {dataset.fp_type} | "
            f"Dimension: {dataset.fp_dim}"
        )

        print("\nAccessing the first sample:")
        anchor_data, pos_data, neg_datas = dataset[0]

        print(
            f"Anchor SMILES: {anchor_data.smi}, "
            f"pLC50: {anchor_data.mol_md.item():.3f}"
        )
        print(f"  mol_fp shape: {anchor_data.mol_fp.shape}")

        print(
            f"Positive SMILES: {pos_data.smi}, "
            f"pLC50: {pos_data.mol_md.item():.3f}"
        )

        print(f"Number of negative samples: {len(neg_datas)}")

        for i, neg_data in enumerate(neg_datas):
            print(
                f" Negative {i + 1} SMILES: {neg_data.smi}, "
                f"pLC50: {neg_data.mol_md.item():.3f}"
            )

    except FileNotFoundError as e:
        print(e)
