from rdkit import Chem
import os
import torch
import copy
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import Batch

from MTP_GAT.utils.data import (
    mol_to_graph_data_obj_rich,
    mol_to_graph_data_obj_super_rich,
    onek_encoding_unk,
    atom_to_vocab,
    allowable_features,
    get_scaffold
)


class MoleculeDataset(Dataset):
    def __init__(
        self,
        data_dir,
        mol_sim_data_file='pretain.csv',
        num_candidates=3,
        feat_type='rich',
        fp_type='ecfp4',
        fp_dim=1024
    ):

        self.data_dir = data_dir
        self.mol_sim_data_file = mol_sim_data_file
        self.num_candidates = num_candidates
        self.feat_type = feat_type
        self.fp_type = fp_type.lower()

        processed_file_name = (
            f"{os.path.splitext(mol_sim_data_file)[0]}_{feat_type}_{fp_type}.pt"
        )
        processed_path = os.path.join(
            data_dir, 'processed_cache', processed_file_name
        )

        if not os.path.exists(processed_path):
            raise FileNotFoundError(
                f"Preprocessed file not found: {processed_path}. "
                f"Please run 'preprocess_molecules.py' first."
            )

        print(f"Loading preprocessed data: {processed_path}")
        self.data_list = torch.load(processed_path)

        print(f"Successfully loaded {len(self.data_list)} preprocessed molecules.")

        if self.feat_type == 'rich':
            self.node_feature_dims = 144
            self.edge_feature_dims = 14
        elif self.feat_type == 'super_rich':
            self.node_feature_dims = 164
            self.edge_feature_dims = 13
        else:
            raise NotImplementedError(
                f"Unsupported feature type: {self.feat_type}"
            )

        self.mask_node_token = self._build_mask_node_token()
        self.mask_edge_token = self._build_mask_edge_token()

    def _build_mask_node_token(self):

        if self.feat_type == 'rich':
            mask_token = (
                onek_encoding_unk(0, list(range(1, 101))) +
                onek_encoding_unk(0, list(range(11))) +
                onek_encoding_unk(0, list(range(-2, 3))) +
                onek_encoding_unk(0, list(range(5))) +
                onek_encoding_unk(
                    Chem.rdchem.HybridizationType.UNSPECIFIED,
                    [
                        Chem.rdchem.HybridizationType.SP,
                        Chem.rdchem.HybridizationType.SP2,
                        Chem.rdchem.HybridizationType.SP3,
                        Chem.rdchem.HybridizationType.SP3D,
                        Chem.rdchem.HybridizationType.SP3D2,
                        Chem.rdchem.HybridizationType.UNSPECIFIED
                    ]
                ) +
                [0] +
                onek_encoding_unk(0, list(range(5))) +
                [0] +
                [0, 0, 0] +
                [0.0]
            )

        elif self.feat_type == 'super_rich':
            mask_token = (
                onek_encoding_unk(0, list(range(119))) +
                onek_encoding_unk(0, [0, 1, 2, 3, 4, 5]) +
                onek_encoding_unk(0, [-1, -2, 1, 2, 0]) +
                onek_encoding_unk(0, [0, 1, 2, 3]) +
                onek_encoding_unk(0, [0, 1, 2, 3, 4]) +
                onek_encoding_unk(
                    Chem.rdchem.HybridizationType.UNSPECIFIED,
                    [
                        Chem.rdchem.HybridizationType.SP,
                        Chem.rdchem.HybridizationType.SP2,
                        Chem.rdchem.HybridizationType.SP3,
                        Chem.rdchem.HybridizationType.SP3D,
                        Chem.rdchem.HybridizationType.SP3D2,
                        Chem.rdchem.HybridizationType.UNSPECIFIED
                    ]
                ) +
                [0] +
                [0.0] +
                onek_encoding_unk(0, [0, 1, 2, 3, 4, 5, 6]) +
                [0] +
                [0] +
                [0] +
                [0] +
                [0, 0, 0, 0, 0, 0]
            )

        else:
            raise NotImplementedError(
                f"Unsupported node feature type: {self.feat_type}"
            )

        mask_token = torch.tensor(mask_token, dtype=torch.long)
        return mask_token

    def _build_mask_edge_token(self):

        if self.feat_type == 'rich':
            mask_token = (
                onek_encoding_unk(
                    Chem.rdchem.BondType.UNSPECIFIED,
                    allowable_features['possible_bonds']
                ) +
                [0] +
                [0] +
                onek_encoding_unk(
                    Chem.rdchem.BondStereo.STEREONONE,
                    allowable_features['possible_bond_stereo']
                )
            )

        elif self.feat_type == 'super_rich':
            mask_token = (
                onek_encoding_unk(
                    Chem.rdchem.BondType.UNSPECIFIED,
                    allowable_features['possible_bonds']
                ) +
                onek_encoding_unk(0, list(range(6))) +
                [0] +
                [0]
            )

        else:
            raise NotImplementedError(
                f"Unsupported edge feature type: {self.feat_type}"
            )

        mask_token = torch.tensor(mask_token, dtype=torch.long)
        return mask_token

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        ori_data = self.data_list[idx]

        candidates = [
            copy.deepcopy(ori_data)
            for _ in range(self.num_candidates + 2)
        ]

        for i in range(2, len(candidates)):
            candidate = candidates[i]
            mask_inds = candidate.mask_inds

            if len(mask_inds) > 0:
                candidate.x[mask_inds] = self.mask_node_token

                mask_edge_inds = set()

                for node_idx in mask_inds:
                    edge_indices_from = (
                        candidate.edge_index[0] == node_idx
                    ).nonzero(as_tuple=True)[0]

                    mask_edge_inds.update(edge_indices_from.tolist())

                    edge_indices_to = (
                        candidate.edge_index[1] == node_idx
                    ).nonzero(as_tuple=True)[0]

                    mask_edge_inds.update(edge_indices_to.tolist())

                if mask_edge_inds:
                    candidate.edge_attr[list(mask_edge_inds)] = (
                        self.mask_edge_token
                    )

        for candidate in candidates:
            if hasattr(candidate, 'mask_inds'):
                del candidate.mask_inds

        return candidates


def collate_fn(batch):

    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    flat_batch = [
        item for sublist in batch for item in sublist
    ]

    return Batch.from_data_list(flat_batch)


if __name__ == '__main__':

    try:
        BASE_DIR = ''
        DATA_DIR = os.path.join(BASE_DIR, 'data/pretrain')

        dataset = MoleculeDataset(
            data_dir=DATA_DIR,
            mol_sim_data_file='pretain.csv',
            feat_type='rich',
            fp_type='ecfp4'
        )

        from torch.utils.data import DataLoader

        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_fn
        )

        print("\n--- Testing the optimized data loader ---")

        for i, batch in enumerate(loader):

            if batch is None:
                print("Skipping empty batch...")
                continue

            print(f"Batch {i + 1} loaded successfully!")
            print(f"Number of graphs in batch: {batch.num_graphs}")
            print(f"Node feature shape: {batch.x.shape}")
            print(f"Edge feature shape: {batch.edge_attr.shape}")

            print(
                f"Node mask token data type: "
                f"{dataset.mask_node_token.dtype}"
            )
            print(f"Node feature data type: {batch.x.dtype}")

            if i >= 1:
                break

    except FileNotFoundError as e:
        print(e)
        print(
            "\nPlease run 'preprocess_molecules.py' first "
            "to generate the preprocessed data."
        )