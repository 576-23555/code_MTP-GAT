import os
import csv
from datetime import datetime
import numpy as np
import yaml
import torch
import torch.optim as optim
import torch_geometric
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from MTP_GAT.pretrain.loader_molsim import MoleculeDataset
from MTP_GAT.pretrain.loader_ActiveCliffs import ActiveCliffsDataset
from MTP_GAT.pretrain.model import GNNWrapper
from MTP_GAT.gat import GAT


def collate_mol_sim_batch(batch):
    try:
        flat_batch = []
        all_mol_fps_for_batch = []  
        for i, data_tuple in enumerate(batch):
            if not isinstance(data_tuple, (list, tuple)) or len(data_tuple) < 2:
                print(f"Warning: MolSim batch sample {i} has an invalid structure; expected a list or tuple containing at least anchor and positive. Actual type: {type(data_tuple)}, length: {len(data_tuple) if hasattr(data_tuple, '__len__') else 'N/A'}. Skipping this sample.")
                continue

            anchor_data = data_tuple[0]
            pos_data = data_tuple[1]
            random_neg_datas_list = data_tuple[2:]

            flat_batch.append(anchor_data)
            all_mol_fps_for_batch.append(anchor_data.mol_fp)                   

            flat_batch.append(pos_data)
            all_mol_fps_for_batch.append(pos_data.mol_fp)                   

            if isinstance(random_neg_datas_list, (list, tuple)):
                for neg_data in random_neg_datas_list:
                    flat_batch.append(neg_data)
                    all_mol_fps_for_batch.append(neg_data.mol_fp)                  
            else:
                flat_batch.append(random_neg_datas_list)
                all_mol_fps_for_batch.append(random_neg_datas_list.mol_fp)                  

        valid_flat_batch = []
        valid_mol_fps_for_batch = []                        
        for i, data in enumerate(flat_batch):
            if not isinstance(data, torch_geometric.data.Data):
                print(f"Warning: MolSim sample {i} (flat_batch) is not a valid PyG Data object. Type: {type(data)}. Skipping this sample.")
                continue

            if not hasattr(data, 'x') or not hasattr(data, 'edge_index'):
                print(f"Warning: MolSim sample {i} is missing x or edge_index. SMILES: {getattr(data, 'smi', 'N/A')}. Skipping this sample.")
                continue

            if data.x.size(0) == 0:
                print(f"Warning: MolSim sample {i} has 0 nodes. SMILES: {getattr(data, 'smi', 'N/A')}. Skipping this sample.")
                continue

            if data.edge_index.numel() > 0:
                if data.edge_index.max().item() >= data.x.size(0) or data.edge_index.min().item() < 0:
                    print(f"Warning: MolSim sample {i} has edge indices out of range or below zero. x.size(0)={data.x.size(0)}, edge_index min/max={data.edge_index.min().item()}/{data.edge_index.max().item()}. SMILES: {getattr(data, 'smi', 'N/A')}. Skipping this sample.")
                    continue

            valid_flat_batch.append(data)
                                        
            valid_mol_fps_for_batch.append(all_mol_fps_for_batch[i])

        if not valid_flat_batch:
            print("Warning: MolSim collate_fn received an empty or all-invalid batch. Returning None.")
            return None

        batch_output = Batch.from_data_list(valid_flat_batch)
        batch_output.mol_fps = valid_mol_fps_for_batch                                   
        return batch_output
    except Exception as e:
        print(f"MolSim collation error: {type(e).__name__}: {e}")
        print("Printing original batch details for debugging:")
        for i, data_item in enumerate(batch):
            print(f" Original Batch Item {i}: Type={type(data_item)}, Length={len(data_item) if hasattr(data_item, '__len__') else 'N/A'}")
            if hasattr(data_item, '__len__'):
                for j, sub_item in enumerate(data_item):
                    smi_val = getattr(sub_item, 'smi', 'N/A')
                    nodes_val = getattr(sub_item, 'x', torch.empty(0)).size(0) if hasattr(sub_item, 'x') else 'N/A'
                    print(f" Sub-item {j}: Type={type(sub_item)}, SMILES={smi_val}, Nodes={nodes_val}, x_shape={getattr(sub_item, 'x', torch.empty(0)).shape}")
        raise

def collate_act_cliff_batch(batch):
    try:
        flat_batch = []
        all_mol_fps_for_batch = []

        for i, data_tuple in enumerate(batch):
            anchor_data, pos_data, neg_datas = data_tuple

            flat_batch.append(anchor_data)
            all_mol_fps_for_batch.append(anchor_data.mol_fps)

            flat_batch.append(pos_data)
            all_mol_fps_for_batch.append(pos_data.mol_fps)

            for neg_data in neg_datas:
                flat_batch.append(neg_data)
                all_mol_fps_for_batch.append(neg_data.mol_fps)

        valid_flat_batch = []
        valid_mol_fps_for_batch = []

        for i, data in enumerate(flat_batch):
            if not isinstance(data, Data) or \
               not hasattr(data, 'x') or data.x.size(0) == 0 or \
               not hasattr(data, 'edge_index'):
                continue


            if data.edge_index.numel() > 0:
                if data.edge_index.max().item() >= data.x.size(0) or data.edge_index.min().item() < 0:
                    continue

            valid_flat_batch.append(data)
            if i < len(all_mol_fps_for_batch):
                valid_mol_fps_for_batch.append(all_mol_fps_for_batch[i])

        if not valid_flat_batch:
            print("WARNING: ActCliff collate_fn resulted in an empty valid_flat_batch. Returning None.")
            return None

        batch_output = Batch.from_data_list(valid_flat_batch)
        batch_output.mol_fps = valid_mol_fps_for_batch

        return batch_output
    except Exception as e:
        print(f"ERROR: ActCliff batch collation failed: {type(e).__name__}: {e}")
        raise


def train_one_epoch(model, mol_sim_loader, act_cliff_loader, optimizer, config):
    model.train()
    accum_iter = config['optim']['accum_iter']

    loss_history = {
        'loss_1': [],
        'loss_1_reg': [],
        'loss_2': [],
        'loss_2_reg': [],
        'loss_total': []
    }

    mol_sim_iter = iter(mol_sim_loader)
    act_cliff_iter = iter(act_cliff_loader)

    total_steps = len(mol_sim_loader)
    pbar = tqdm(range(total_steps), desc='Training')

    optimizer.zero_grad()

    for step_idx in pbar:
        current_total_loss = 0.0
        current_loss_1 = 0.0
        current_loss_1_reg = 0.0
        current_loss_2 = 0.0
        current_loss_2_reg = 0.0


        mol_sim_batch = None
        try:
            mol_sim_batch = next(mol_sim_iter)
            if mol_sim_batch is not None:
                mol_sim_batch = mol_sim_batch.to(config['device'])
        except StopIteration:
            mol_sim_iter = iter(mol_sim_loader)
            mol_sim_batch = next(mol_sim_iter)
            if mol_sim_batch is not None:
                mol_sim_batch = mol_sim_batch.to(config['device'])
        except Exception as e:
            print(f"Error getting MolSim batch at step {step_idx}: {e}")
            mol_sim_batch = None

        if mol_sim_batch is not None:
            if step_idx == 0:
                print(f"- **mol_sim_batch.mol_fps type: {type(mol_sim_batch.mol_fps)}**")
                if hasattr(mol_sim_batch, 'mol_fps') and isinstance(mol_sim_batch.mol_fps, list):
                    if len(mol_sim_batch.mol_fps) > 0:
                        print(f"- **mol_sim_batch.mol_fps[0] shape: {mol_sim_batch.mol_fps[0].shape}**")


            with torch.set_grad_enabled(True):
                mol_sim_output = model(mol_sim_batch, task_type='molsim')

                loss_1_val = torch.nan_to_num(mol_sim_output['loss_1'])
                loss_1_reg_val = torch.nan_to_num(mol_sim_output['loss_1_reg'])
                reg_coeff_1 = config['optim'].get('reg_coeff_molsim', 0.1)

                mol_sim_loss = (loss_1_val + reg_coeff_1 * loss_1_reg_val)
                current_total_loss += mol_sim_loss
                current_loss_1 = float(loss_1_val.item())
                current_loss_1_reg = float(loss_1_reg_val.item())

                loss_history['loss_1'].append(current_loss_1)
                loss_history['loss_1_reg'].append(current_loss_1_reg)

                (mol_sim_loss / accum_iter).backward()
        else:
            loss_history['loss_1'].append(0.0)
            loss_history['loss_1_reg'].append(0.0)


        act_cliff_batch = None
        try:
            act_cliff_batch = next(act_cliff_iter)
            if act_cliff_batch is not None:
                act_cliff_batch = act_cliff_batch.to(config['device'])
        except StopIteration:
            act_cliff_iter = iter(act_cliff_loader)
            act_cliff_batch = next(act_cliff_iter)
            if act_cliff_batch is not None:
                act_cliff_batch = act_cliff_batch.to(config['device'])
        except Exception as e:
            print(f"Error getting ActCliff batch at step {step_idx}: {e}")
            act_cliff_batch = None

        if act_cliff_batch is not None:
            if step_idx == 0:
                print(f"- **act_cliff_batch.mol_fps type: {type(act_cliff_batch.mol_fps)}**")
                if hasattr(act_cliff_batch, 'mol_fps') and isinstance(act_cliff_batch.mol_fps, list):
                    if len(act_cliff_batch.mol_fps) > 0:
                        print(f"- **act_cliff_batch.mol_fps[0] shape: {act_cliff_batch.mol_fps[0].shape}**")


            with torch.set_grad_enabled(True):
                act_cliff_output = model(act_cliff_batch, task_type='actcliff')

                loss_2_val = torch.nan_to_num(act_cliff_output['loss_2'])
                loss_2_reg_val = torch.nan_to_num(act_cliff_output['loss_2_reg'])
                reg_coeff_2 = config['optim'].get('reg_coeff_actcliff', 0.1)

                act_cliff_loss = (loss_2_val + reg_coeff_2 * loss_2_reg_val)
                current_total_loss += act_cliff_loss
                current_loss_2 = float(loss_2_val.item())
                current_loss_2_reg = float(loss_2_reg_val.item())

                loss_history['loss_2'].append(current_loss_2)
                loss_history['loss_2_reg'].append(current_loss_2_reg)

                (act_cliff_loss / accum_iter).backward()
        else:
            loss_history['loss_2'].append(0.0)
            loss_history['loss_2_reg'].append(0.0)

        if mol_sim_batch is not None or act_cliff_batch is not None:
            loss_history['loss_total'].append(float(current_total_loss.item()))
        else:
            loss_history['loss_total'].append(0.0)

        if ((step_idx + 1) % accum_iter == 0) or (step_idx + 1 == total_steps):
            optimizer.step()
            optimizer.zero_grad()

        pbar.set_postfix({
            'L1': f'{current_loss_1:.4f}',
            'L1_reg': f'{current_loss_1_reg:.4f}',
            'L2': f'{current_loss_2:.4f}',
            'L2_reg': f'{current_loss_2_reg:.4f}',
            'Total': f'{current_total_loss.item():.4f}'
        })

    return loss_history


def main(config):
    print(config)                                     
    backbone_name = config['model'].get('name', 'gat')
    device = torch.device(config['device'])
    torch.backends.cudnn.benchmark = True

    data_dir = os.path.abspath(config['dataset']['data_dir'])
    print(f"\nData directory: {data_dir}")
    try:
        print(f"Directory contents: {os.listdir(data_dir)}")
    except FileNotFoundError:
        print(f"Warning: Data directory {data_dir} does not exist.")

    required_files = [
        'atom_vocab_pretrain.pkl',
        'pLC50.csv',
        config['dataset'].get('mol_sim_data_file', 'pretain.csv')
    ]

    for file in required_files:
        full_path = os.path.join(data_dir, file)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Required file not found: {full_path}")

    print("\nInitializing datasets and data loaders...")


    print("Initializing MoleculeDataset for MolSim...")
    mol_sim_dataset = MoleculeDataset(
        data_dir=data_dir,
        mol_sim_data_file=config['dataset'].get('mol_sim_data_file', 'pretain.csv'),
        num_candidates=config['optim'].get('num_candidates', 3),
        feat_type=config['dataset']['feat_type'],
                                                               
        fp_type=config['model'].get('fp_type', 'ecfp4'),
        
        fp_dim=config['model'].get('fp_dim', None)
    )
    print(f"MoleculeDataset size: {len(mol_sim_dataset)}")

    mol_sim_loader = DataLoader(
        mol_sim_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=min(config['dataset']['num_workers'], os.cpu_count() - 1 if os.cpu_count() > 1 else 0),
        collate_fn=collate_mol_sim_batch,
        pin_memory=True,
        drop_last=True
    )
    print(f"MolSim DataLoader created with {len(mol_sim_loader)} batches.")


    act_cliff_dataset = ActiveCliffsDataset(
        data_dir=data_dir,
        data_file='pLC50.csv',
        feat_type=config['dataset']['feat_type'],
        fp_type=config['model'].get('fp_type', 'ecfp4'),
        fp_dim=config['model'].get('fp_dim', None),
    )
    print(f"ActiveCliffsDataset size: {len(act_cliff_dataset)}")

    if not hasattr(act_cliff_dataset, 'atom_vocab_itos') or not act_cliff_dataset.atom_vocab_itos:
        raise RuntimeError("atom_vocab_itos could not be loaded. Check atom_vocab_pretrain.pkl.")

    print(f"Atom vocabulary size: {len(act_cliff_dataset.atom_vocab_itos)}")

    act_cliff_loader = DataLoader(
        act_cliff_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=min(config['dataset']['num_workers'], os.cpu_count() - 1 if os.cpu_count() > 1 else 0),
        collate_fn=collate_act_cliff_batch,
        pin_memory=True,
        drop_last=True
    )
    print(f"ActCliff DataLoader created with {len(act_cliff_loader)} batches.")


    print("\nInitializing model...")

    atom_feat_dim, bond_feat_dim = None, None
    if config['dataset']['feat_type'] == 'basic':
        atom_feat_dim, bond_feat_dim = 75, 14
    elif config['dataset']['feat_type'] == 'rich':
        atom_feat_dim, bond_feat_dim = 144, 14
    elif config['dataset']['feat_type'] == 'super_rich':
        atom_feat_dim, bond_feat_dim = 164, 13
    else:
        raise ValueError(f"Unknown feature type: {config['dataset']['feat_type']}")

    print(f"Feature dimensions for 'feat_type': '{config['dataset']['feat_type']}', atom feature dimension atom_feat_dim={atom_feat_dim}, bond feature dimension bond_feat_dim={bond_feat_dim}")


    backbone_name = config['model'].get('name', 'gat')
    if backbone_name.lower() != 'gat':
        raise ValueError("This training script supports only the GAT backbone.")

    gnn_params = config['model'].get('gnn_params_gat', {})
    gnn = GAT(
        in_feats=gnn_params.get('in_feats', atom_feat_dim),
        hidden_feats=gnn_params.get('hidden_feats', 64),
        out_feats=gnn_params.get('out_feats', 16),
        num_layers=gnn_params.get('num_layers', 2),
        num_heads=gnn_params.get('num_heads', 4),
        dropout=gnn_params.get('dropout', 0.4),
        alpha=gnn_params.get('alpha', 0.2),
        final_out_dim=gnn_params.get('final_out_dim', 1),
        use_norm=gnn_params.get('use_norm', False),
        norm_type=gnn_params.get('norm_type', 'layernorm')
    )
    emb_dim = gnn_params.get('out_feats', 16) * gnn_params.get('num_heads', 4)

    atom_context_size_from_loader = len(act_cliff_dataset.atom_vocab_itos)
    fp_dim_from_loader = act_cliff_dataset.fp_dim


    model = GNNWrapper(
        gnn,
        emb_dim,
        config,
        atom_context_size=atom_context_size_from_loader,
        fp_dim=fp_dim_from_loader
    ).to(device)


    print(f"Initialized {backbone_name} model. Final GNN embedding dimension: {emb_dim}")
    print(model)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config['optim']['lr']),
        weight_decay=float(config['optim']['decay']),
        eps=1e-6
    )

    writer = SummaryWriter(log_dir=os.path.join(config['output_dir'], 'runs'))
    print(f"\nTensorBoard logs will be saved to: {os.path.join(config['output_dir'], 'runs')}")

    loss_csv_path = os.path.join(config['output_dir'], 'training_loss.csv')
    csv_file_exists = os.path.exists(loss_csv_path)

    csv_file = open(loss_csv_path, 'a', newline='')
    csv_writer = csv.writer(csv_file)

    if not csv_file_exists or os.path.getsize(loss_csv_path) == 0:
        csv_writer.writerow(['Epoch', 'Average_Loss_1', 'Average_Loss_1_Reg', 'Average_Loss_2', 'Average_Loss_2_Reg', 'Average_Total_Loss'])
        print(f"Created or initialized loss log CSV file: {loss_csv_path}")
    else:
        print(f"Appending losses to existing CSV file: {loss_csv_path}")

    print("\nStarting training...")
    best_loss = float('inf')
    for epoch in range(config['start_epoch'] + 1, config['epochs'] + 1):
        try:
            loss_history = train_one_epoch(model, mol_sim_loader, act_cliff_loader, optimizer, config)
            avg_loss_1 = np.mean(loss_history['loss_1']) if loss_history['loss_1'] else 0.0
            avg_loss_1_reg = np.mean(loss_history['loss_1_reg']) if loss_history['loss_1_reg'] else 0.0
            avg_loss_2 = np.mean(loss_history['loss_2']) if loss_history['loss_2'] else 0.0
            avg_loss_2_reg = np.mean(loss_history['loss_2_reg']) if loss_history['loss_2_reg'] else 0.0
            avg_total_loss = np.mean(loss_history['loss_total']) if loss_history['loss_total'] else 0.0


            print(f"\nEpoch {epoch} - Average Total Loss: {avg_total_loss:.4f}, "
                             f"Average Loss_1 (MolSim): {avg_loss_1:.4f}, "
                             f"Average Loss_1_Reg: {avg_loss_1_reg:.4f}, "
                             f"Average Loss_2 (ActCliff): {avg_loss_2:.4f}, "
                             f"Average Loss_2_Reg: {avg_loss_2_reg:.4f}")

            writer.add_scalar('Loss/train_total', avg_total_loss, epoch)
            writer.add_scalar('Loss/train_loss_1', avg_loss_1, epoch)
            writer.add_scalar('Loss/train_loss_1_reg', avg_loss_1_reg, epoch)
            writer.add_scalar('Loss/train_loss_2', avg_loss_2, epoch)
            writer.add_scalar('Loss/train_loss_2_reg', avg_loss_2_reg, epoch)

            csv_writer.writerow([epoch, f"{avg_loss_1:.4f}", f"{avg_loss_1_reg:.4f}",
                                        f"{avg_loss_2:.4f}", f"{avg_loss_2_reg:.4f}", f"{avg_total_loss:.4f}"])
            csv_file.flush()

            if avg_total_loss < best_loss:
                best_loss = avg_total_loss
                ckpt = os.path.join(config['output_dir'], f"{config['method']}-best.pt")
                torch.save({'wrapper': model.state_dict()}, ckpt)
                print(f"Saved best model: {ckpt}")

            if epoch % config.get('save_interval', 2) == 0:
                ckpt = os.path.join(config['output_dir'], f"{config['method']}-ep{epoch}.pt")
                torch.save({'wrapper': model.state_dict()}, ckpt)

        except Exception as e:
            print(f"\nEpoch {epoch} Training failed: {type(e).__name__}: {e}")
            if 'CUDA out of memory' in str(e):
                torch.cuda.empty_cache()
                print("CUDA cache cleared. Continuing training.")
                continue
            else:
                csv_file.close()
                raise


    writer.close()
    csv_file.close()


if __name__ == '__main__':
    config_path = 'config/pretrain.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config['dataset']['data_dir'] = os.path.abspath(config['dataset']['data_dir'])
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    original_output_dir = config.get('output_dir', 'checkpoints')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir_name = f"{config['method']}_{timestamp}"
    new_output_dir = os.path.join(original_output_dir, run_dir_name)

    config['output_dir'] = new_output_dir

    os.makedirs(config['output_dir'], exist_ok=True)
    print(f"\nTraining outputs will be saved to: {config['output_dir']}")

    main(config)