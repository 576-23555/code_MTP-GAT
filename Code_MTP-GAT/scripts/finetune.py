import os
import sys
import glob
import copy
import random
import pickle
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from sklearn.metrics import r2_score, mean_squared_error
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from MTP_GAT.finetune.loader import MoleculeDataset
from MTP_GAT.finetune.model import GNNPredictor
from MTP_GAT.finetune.prompt_optim import optimize_prompt_weight_ri as optimize_prompt_weight_ri_
from MTP_GAT.utils.scheduler import PolynomialDecayLR


def resolve_checkpoint_path(config):
    if config['model']['checkpoint'] and os.path.exists(config['model']['checkpoint']):
        return os.path.abspath(config['model']['checkpoint'])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(script_dir, "scripts", "checkpoint")

    if os.path.exists(checkpoint_dir):
        all_checkpoints = []

        for pattern in ["*.pt", "**/*.pt"]:
            all_checkpoints.extend(
                glob.glob(os.path.join(checkpoint_dir, pattern), recursive=True)
            )

        if all_checkpoints:
            valid_checkpoints = [
                p for p in all_checkpoints if os.path.getsize(p) > 0
            ]

            if valid_checkpoints:
                valid_checkpoints.sort(
                    key=lambda p: os.path.getmtime(p),
                    reverse=True
                )
                print(
                    f"Could not find the checkpoint at the specified path. "
                    f"Using the latest valid checkpoint: {valid_checkpoints[0]}"
                )
                return os.path.abspath(valid_checkpoints[0])
            else:
                print(
                    f"Checkpoint files were found in '{checkpoint_dir}', "
                    "but all files appear to be empty."
                )

    raise FileNotFoundError(
        "No valid checkpoint file was found. Check config['model']['checkpoint'] "
        "or ensure that a valid checkpoint exists in 'scripts/checkpoint'."
    )


def get_optimizer(model, lr_params):
    assert isinstance(lr_params, dict)

    pretrain_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and ('gnn' in n or 'aggrs' in n)
    ]

    finetune_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and 'graph_pred_linear' in n
    ]

    prompt_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and ('prompt_weight' in n or 'moe_gate' in n)
    ]

    other_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and not any(
            key in n for key in [
                'gnn',
                'aggrs',
                'graph_pred_linear',
                'prompt_weight',
                'moe_gate'
            ]
        )
    ]

    finetune_params.extend(other_params)

    params = []

    if finetune_params:
        params.append({
            'params': finetune_params,
            'lr': float(lr_params['finetune_lr'])
        })

    if pretrain_params:
        params.append({
            'params': pretrain_params,
            'lr': float(lr_params['pretrain_lr'])
        })

    if prompt_params:
        params.append({
            'params': prompt_params,
            'lr': float(lr_params['prompt_lr'])
        })

    if not params:
        raise ValueError("No trainable parameters were found.")

    optimizer = torch.optim.Adam(
        params,
        weight_decay=float(lr_params['decay'])
    )

    return optimizer


def get_dataloader_from_split(config):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_dir = os.path.join(
        script_dir,
        '..',
        'data',
        'finetune',
    )
    base_data_dir = os.path.normpath(base_data_dir)

    split_dir = os.path.join(base_data_dir, 'split-ERα')

    train_file = os.path.join(split_dir, 'train_split.csv')
    test_file = os.path.join(split_dir, 'test_split.csv')

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")

    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    train_dataset = MoleculeDataset(
        data_dir=split_dir,
        data_name='train_split',
        feat_type=config['dataset']['feat_type']
    )

    test_dataset = MoleculeDataset(
        data_dir=split_dir,
        data_name='test_split',
        feat_type=config['dataset']['feat_type']
    )

    print("--- Using predefined dataset splits ---")
    print(f"Training set size: {len(train_dataset)}, Test set size: {len(test_dataset)}")
    print(f"Training file: {train_file}")
    print(f"Test file: {test_file}")

    num_workers = min(
        config['dataset']['num_workers'],
        os.cpu_count() - 1 if os.cpu_count() > 1 else 0
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataset, train_loader, None, test_loader


def split_train_val(train_dataset, config, seed=42):
    val_ratio = float(config.get('finetune', {}).get('val_ratio', 0.1))

    num_workers = min(
        config['dataset']['num_workers'],
        os.cpu_count() - 1 if os.cpu_count() > 1 else 0
    )

    if val_ratio <= 0 or len(train_dataset) < 50:
        print(
            "Validation set disabled because val_ratio <= 0 "
            "or the dataset contains too few samples. "
            "The test set will be used for model selection."
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True
        )

        return train_loader, None

    labels = np.array([
        float(np.ravel(l)[0]) for l in train_dataset.labels
    ])

    idx = np.arange(len(labels))
    n_bins = int(min(5, max(2, len(labels) // 20)))

    try:
        bins = pd.qcut(
            labels,
            q=n_bins,
            labels=False,
            duplicates='drop'
        )
    except Exception:
        bins = np.zeros_like(labels, dtype=int)

    train_idx, val_idx = train_test_split(
        idx,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=bins
    )

    print(
        f"Training/validation split: "
        f"train {len(train_idx)} / val {len(val_idx)}"
    )

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        Subset(train_dataset, val_idx),
        batch_size=config['batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def eval(model, loader, config):
    model.eval()

    y_true, y_scores = [], []
    all_attn_weights_list = []
    prompt_scores_list = []

    for step, batch in enumerate(loader):
        batch = batch.to(config['device'])

        with torch.no_grad():
            if config['model']['return_attention_weights']:
                predict, scores, raw_attn, _ = model(
                    batch,
                    return_attention_weights=True
                )

                all_attn_weights_list.append(raw_attn)
                prompt_scores_list.append(scores)
            else:
                predict, _ = model(batch)

            if isinstance(batch.label, dict):
                label = batch.label['label'].view_as(predict)
            else:
                label = batch.label.view_as(predict)

            is_valid = ~torch.isnan(label)

            y_true.append(label[is_valid])
            y_scores.append(predict[is_valid])

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    metrics = {}

    if len(y_true) > 0:
        mse = mean_squared_error(y_true, y_scores)
        metrics['rmse'] = np.sqrt(mse)
        metrics['r2'] = r2_score(y_true, y_scores)
    else:
        metrics['rmse'] = float('nan')
        metrics['r2'] = float('nan')

    if config['model']['return_attention_weights']:
        return metrics, all_attn_weights_list, prompt_scores_list

    return metrics, None, None


def train(model, train_loader, criterion, optimizer, scheduler, config):
    model.train()

    loss_history = []
    y_true_train, y_scores_train = [], []

    for idx, batch in enumerate(train_loader):
        batch = batch.to(config['device'])

        predict, _ = model(batch)

        if isinstance(batch.label, dict):
            label = batch.label['label'].view_as(predict)
        else:
            label = batch.label.view_as(predict)

        is_valid = ~torch.isnan(label)

        if is_valid.sum() == 0:
            continue

        loss = criterion(predict[is_valid], label[is_valid])
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()

        if config['optim']['gradient_clip'] > 0:
            nn.utils.clip_grad_norm_(
                model.parameters(),
                config['optim']['gradient_clip']
            )

        optimizer.step()

        if config['optim']['scheduler'] == 'poly_decay':
            scheduler.step()

        loss_history.append(loss.item())

        y_true_train.append(label[is_valid].detach().cpu())
        y_scores_train.append(predict[is_valid].detach().cpu())

    y_true_train_np = torch.cat(y_true_train, dim=0).numpy()
    y_scores_train_np = torch.cat(y_scores_train, dim=0).numpy()

    train_metrics = {}

    if len(y_true_train_np) > 0:
        mse_train = mean_squared_error(
            y_true_train_np,
            y_scores_train_np
        )
        train_metrics['rmse'] = np.sqrt(mse_train)
        train_metrics['r2'] = r2_score(
            y_true_train_np,
            y_scores_train_np
        )
    else:
        train_metrics['rmse'] = float('nan')
        train_metrics['r2'] = float('nan')

    return np.mean(loss_history), train_metrics


def optimize_prompt_weight_ri(
    model,
    train_loader,
    val_loader,
    config,
    metric='euclidean',
    act='softmax',
    max_num=5000
):
    temperature = config['model']['temperature']
    prompt_cfg = config.get('prompt_optim', {})
    skip_bo = prompt_cfg.get('skip_bo', False)

    num = 0
    model.eval()

    graph_rep_list, label_list = [], []

    for batch in tqdm(
        train_loader,
        desc="Collecting graph representations",
        leave=False
    ):
        batch = batch.to(config['device'])

        with torch.no_grad():
            graph_reps_batch, _, _, _ = model.get_representations(
                batch,
                channel_idx=-1
            )

            if isinstance(batch.label, dict):
                labels_batch_raw = batch.label['label'].view(
                    -1,
                    model.num_tasks
                )
            else:
                labels_batch_raw = batch.label.view(
                    -1,
                    model.num_tasks
                )

            is_valid = ~torch.isnan(labels_batch_raw).any(dim=1)

            if is_valid.sum() == 0:
                continue

            graph_rep_list.append(
                graph_reps_batch[:, is_valid, :]
            )

            label_list.append(labels_batch_raw[is_valid])

            num += is_valid.sum().item()

            if num > max_num:
                break

    if not graph_rep_list:
        raise ValueError(
            "No valid samples are available for prompt weight optimization."
        )

    graph_reps = torch.cat(graph_rep_list, dim=1).cpu()
    labels = torch.cat(label_list, dim=0).cpu()

    print(
        f"Collected {graph_reps.shape[1]} valid samples "
        f"for MoE optimization. Graph representations: "
        f"{tuple(graph_reps.shape)}"
    )

    return optimize_prompt_weight_ri_(
        graph_reps,
        labels,
        n_runs=prompt_cfg.get('n_runs', 50),
        n_inits=prompt_cfg.get('n_inits', 50),
        n_points=prompt_cfg.get('n_points', 5),
        n_restarts=prompt_cfg.get('n_restarts', 512),
        n_samples=prompt_cfg.get('n_samples', 512),
        temperature=temperature,
        metric=metric,
        skip_bo=skip_bo,
        verbose=config.get('verbose', True),
        cliff_delta=prompt_cfg.get('cliff_delta', 1.0),
        noncliff_delta=prompt_cfg.get('noncliff_delta', 0.2),
        max_pairs=prompt_cfg.get('max_pairs', 20000),
    )


def mean_gate_weights(model, loader, config):
    if not getattr(model, 'use_moe', False):
        return None

    model.eval()
    probs_all = []

    for batch in loader:
        batch = batch.to(config['device'])

        with torch.no_grad():
            probs = model.get_moe_weights(batch)

        probs_all.append(probs.detach().cpu())

    if not probs_all:
        return None

    return torch.cat(probs_all, dim=0).mean(dim=0).numpy().tolist()


def main(config):
    if config['batch_size'] % 2 != 0:
        config['batch_size'] = max(2, config['batch_size'] - 1)
        print(
            f"Adjusted batch_size to {config['batch_size']} "
            "to ensure an even batch size."
        )

    num_runs = 5
    runseeds = np.random.SeedSequence().generate_state(num_runs)

    print(
        f"\nRandom seeds for five independent runs: "
        f"{runseeds.tolist()}"
    )

    if config['dataset']['feat_type'] == 'basic':
        atom_feat_dim, bond_feat_dim = 75, 14
    elif config['dataset']['feat_type'] == 'rich':
        atom_feat_dim, bond_feat_dim = 143, 14
    elif config['dataset']['feat_type'] == 'super_rich':
        atom_feat_dim, bond_feat_dim = 170, 14
    else:
        raise NotImplementedError(
            "Unsupported feature type. Choose from basic, rich, or super_rich."
        )

    train_dataset, _, _, test_loader = get_dataloader_from_split(config)

    all_run_data = []

    best_initialization_overall = None

    if config['prompt_optim'].get('inits') is not None:
        best_initialization_overall = torch.Tensor(
            config['prompt_optim']['inits']
        )

    backbone_name = 'gat'
    config['model']['name'] = 'gat'

    gnn_params = config['model'].get('gnn_params_gat', {})

    if 'out_feats' not in gnn_params or 'num_heads' not in gnn_params:
        raise ValueError(
            "GAT configuration must contain 'out_feats' and 'num_heads' "
            "under config['model']['gnn_params_gat']."
        )

    emb_dim = gnn_params['out_feats'] * gnn_params['num_heads']

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    result_dir = os.path.join(
        'results',
        f'finetune_run_{timestamp}'
    )

    checkpoint_dir = os.path.join(result_dir, 'checkpoints')
    per_epoch_dir = os.path.join(result_dir, 'per_epoch_metrics')

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(per_epoch_dir, exist_ok=True)

    print(f"\nResults will be saved to: {result_dir}")

    for i in range(num_runs):
        print(
            f"\n--- Running experiment {i + 1}/{num_runs} "
            f"with seed {runseeds[i]} ---"
        )

        set_seed(int(runseeds[i]))

        print("Creating GNNPredictor with emb_dim =", emb_dim)

        train_loader, val_loader = split_train_val(
            train_dataset,
            config,
            seed=int(runseeds[i])
        )

        model = GNNPredictor(
            emb_dim=emb_dim,
            num_tasks=train_dataset.num_task,
            normalize=config['model'].get('normalize', False),
            atom_feat_dim=atom_feat_dim,
            bond_feat_dim=bond_feat_dim,
            drop_ratio=config['model']['dropout_ratio'],
            attn_drop_ratio=config['model'].get(
                'attn_dropout_ratio',
                config['model']['dropout_ratio']
            ),
            temperature=config['model'].get('temperature', 1.0),
            use_prompt=config['model']['use_prompt'],
            model_head=config['model'].get('aggr_head', 4),
            layer_norm_out=config['model'].get('layernorm', False),
            backbone='gat',
            use_moe=config['model'].get('use_moe', True),
            moe_hidden_dim=config['model'].get('moe_hidden_dim', None),
            moe_dropout=config['model'].get('moe_dropout', None),
            moe_learnable=config['model'].get('moe_learnable', True),
            config=config
        )

        if config['model']['checkpoint']:
            try:
                checkpoint_path = resolve_checkpoint_path(config)

                print(f"Checkpoint found: {checkpoint_path}")

                checkpoint_state = torch.load(
                    checkpoint_path,
                    map_location=config['device']
                )['wrapper']

                model_dict = model.state_dict()
                pretrained_dict = {}

                for k, v in checkpoint_state.items():
                    if k in model_dict and model_dict[k].shape == v.shape:
                        pretrained_dict[k] = v

                    elif (
                        k.startswith('gnn.')
                        and 'model.' + k in model_dict
                        and model_dict['model.' + k].shape == v.shape
                    ):
                        pretrained_dict['model.' + k] = v

                    elif (
                        k.startswith('model.gnn.')
                        and k[6:] in model_dict
                        and model_dict[k[6:]].shape == v.shape
                    ):
                        pretrained_dict[k[6:]] = v

                    elif (
                        k.startswith('aggrs.')
                        and 'model.' + k in model_dict
                        and model_dict['model.' + k].shape == v.shape
                    ):
                        pretrained_dict['model.' + k] = v

                    elif (
                        k.startswith('model.aggrs.')
                        and k[6:] in model_dict
                        and model_dict[k[6:]].shape == v.shape
                    ):
                        pretrained_dict[k[6:]] = v

                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict, strict=False)

                print(
                    f"Loaded {len(pretrained_dict)} pretrained parameters."
                )
                print("Checkpoint loading completed.")

            except Exception as e:
                print(f"Checkpoint loading failed: {str(e)}")
                print("Continuing with randomly initialized model weights.")

                config['model']['use_prompt'] = False
                print("use_prompt has been set to False.")

        model.to(config['device'])

        if config['model'].get('freeze_gnn', False):
            model.freeze_gnn_module()
            print("GNN parameters are frozen.")
        else:
            print("GNN parameters are trainable.")

        if config['model'].get('freeze_aggr', False):
            model.freeze_aggr_module()
            print("Aggregation module parameters are frozen.")

        moe_init_probs = None

        if config['model']['use_prompt']:
            print("Optimizing Prompt/MoE weights...")

            if (
                best_initialization_overall is None
                or config['prompt_optim'].get('inits') is None
            ):
                best_initialization_current_run = optimize_prompt_weight_ri(
                    model,
                    train_loader,
                    val_loader,
                    config
                ).to(config['device'])

                if (
                    i == 0
                    and config['prompt_optim'].get('inits') is None
                ):
                    best_initialization_overall = (
                        best_initialization_current_run
                    )

                print("Prompt/MoE weight optimization completed.")

            else:
                best_initialization_current_run = (
                    best_initialization_overall.to(config['device'])
                )

                print("Using the previous optimal Prompt/MoE initialization.")

            model.set_prompt_weight(best_initialization_current_run)

            initial_prompt_probs = (
                model.get_prompt_weight('softmax').data.cpu()
            )

            initial_prompt_weights = (
                model.get_prompt_weight('none').data.cpu()
            )

            moe_init_probs = initial_prompt_probs.numpy().tolist()

            print(
                'Initial Prompt logits:',
                np.round(initial_prompt_weights.numpy(), 4)
            )

            print(
                'Initial Prompt probabilities:',
                np.round(initial_prompt_probs.numpy(), 4)
            )

            print(
                'MoE enabled:',
                bool(getattr(model, 'use_moe', False))
            )

        else:
            print("use_prompt is False. Skipping prompt weight optimization.")

        optimizer = get_optimizer(model, config['optim'])
        scheduler = None

        if config['optim']['scheduler'] == 'cos_anneal':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config['epochs'],
                eta_min=config['optim']['min_lr']
            )

        elif config['optim']['scheduler'] == 'poly_decay':
            scheduler = PolynomialDecayLR(
                optimizer,
                warmup_updates=config['epochs'] * len(train_loader) // 10,
                tot_updates=config['epochs'] * len(train_loader),
                lr=config['optim']['finetune_lr'],
                end_lr=config['optim']['min_lr'],
                power=1
            )

        criterion = nn.MSELoss(reduction='none')

        best_val_r2 = -float('inf')
        best_val_rmse = float('inf')

        best_train_r2_score = -float('inf')
        best_train_rmse_score = float('inf')

        best_epoch = -1
        best_checkpoint = None

        per_epoch_data = []

        config['model']['return_attention_weights'] = False

        for epoch in tqdm(
            range(1, config['epochs'] + 1),
            desc="Epochs"
        ):
            avg_train_loss, train_metrics = train(
                model,
                train_loader,
                criterion,
                optimizer,
                scheduler,
                config
            )

            select_loader = (
                val_loader if val_loader is not None else test_loader
            )

            select_metrics, _, _ = eval(
                model,
                select_loader,
                config
            )

            test_metrics, _, _ = (
                eval(model, test_loader, config)
                if val_loader is not None
                else (select_metrics, None, None)
            )

            epoch_data = {
                'run_id': i + 1,
                'seed': runseeds[i],
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'train_r2': train_metrics['r2'],
                'train_rmse': train_metrics['rmse'],
                'val_r2': select_metrics['r2'],
                'val_rmse': select_metrics['rmse'],
                'test_r2': (
                    test_metrics['r2']
                    if val_loader is not None
                    else np.nan
                ),
                'test_rmse': (
                    test_metrics['rmse']
                    if val_loader is not None
                    else np.nan
                ),
            }

            per_epoch_data.append(epoch_data)

            if config['optim']['scheduler'] == 'cos_anneal':
                scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']

            if select_metrics['r2'] > best_val_r2:
                best_val_r2 = select_metrics['r2']
                best_val_rmse = select_metrics['rmse']

                best_train_r2_score = train_metrics['r2']
                best_train_rmse_score = train_metrics['rmse']

                best_epoch = epoch
                best_checkpoint = copy.deepcopy(model.state_dict())

                tqdm.write(
                    f"New best "
                    f"{'validation' if val_loader is not None else 'test'} "
                    f"R2: {best_val_r2:.4f} "
                    f"(RMSE: {best_val_rmse:.4f}). Model saved."
                )

        if best_checkpoint is not None:
            model.load_state_dict(best_checkpoint)

            checkpoint_file_path = os.path.join(
                checkpoint_dir,
                f'run_{i + 1}_best_model_r2_{best_val_r2:.4f}.pt'
            )

            torch.save(best_checkpoint, checkpoint_file_path)

            print(
                f"Best model weights saved to: {checkpoint_file_path}"
            )

        final_test_metrics, _, _ = eval(
            model,
            test_loader,
            config
        )

        final_gate = mean_gate_weights(
            model,
            test_loader,
            config
        )

        per_epoch_df = pd.DataFrame(per_epoch_data)

        per_epoch_file_path = os.path.join(
            per_epoch_dir,
            f'per_epoch_metrics_run_{i + 1}.csv'
        )

        per_epoch_df.to_csv(
            per_epoch_file_path,
            index=False
        )

        print(
            f"Epoch metrics saved to: {per_epoch_file_path}"
        )

        run_data = {
            'run_id': i + 1,
            'seed': runseeds[i],
            'backbone': 'gat',
            'checkpoint_path': config['model']['checkpoint'],
            'use_prompt': config['model']['use_prompt'],
            'use_moe': bool(getattr(model, 'use_moe', False)),
            'epochs': config['epochs'],
            'best_epoch': best_epoch,

            'emb_dim': emb_dim,
            'feat_type': config['dataset']['feat_type'],
            'dropout_ratio': config['model']['dropout_ratio'],
            'attn_dropout_ratio': config['model'].get(
                'attn_dropout_ratio',
                config['model']['dropout_ratio']
            ),
            'temperature': config['model'].get('temperature', 1.0),
            'aggr_head': config['model'].get('aggr_head', 4),
            'normalize': config['model'].get('normalize', False),
            'layernorm': config['model'].get('layernorm', False),
            'freeze_gnn': config['model'].get('freeze_gnn', False),
            'freeze_aggr': config['model'].get('freeze_aggr', False),
            **gnn_params,

            'finetune_lr': config['optim']['finetune_lr'],
            'pretrain_lr': config['optim']['pretrain_lr'],
            'prompt_lr': config['optim']['prompt_lr'],
            'decay': config['optim']['decay'],
            'scheduler': config['optim']['scheduler'],
            'gradient_clip': config['optim']['gradient_clip'],

            'batch_size': config['batch_size'],
            'val_ratio': config.get('finetune', {}).get(
                'val_ratio',
                0.1
            ),

            'best_train_r2': best_train_r2_score,
            'best_train_rmse': best_train_rmse_score,
            'best_val_r2': best_val_r2,
            'best_val_rmse': best_val_rmse,
            'final_test_r2': final_test_metrics['r2'],
            'final_test_rmse': final_test_metrics['rmse'],
        }

        if config['model']['use_prompt']:
            run_data['final_prompt_weights'] = (
                model.get_prompt_weight('softmax')
                .data.cpu().numpy().tolist()
            )

            run_data['moe_init_probs'] = moe_init_probs
            run_data['moe_avg_gate_on_test'] = final_gate

        all_run_data.append(run_data)

        print(
            f"Best training R2: {best_train_r2_score:.4f}, "
            f"Best training RMSE: {best_train_rmse_score:.4f}"
        )

        print(
            f"Best epoch: {best_epoch} | "
            f"Best validation R2: {best_val_r2:.4f}, "
            f"Best validation RMSE: {best_val_rmse:.4f}"
        )

        print(
            f"Final test R2: {final_test_metrics['r2']:.4f}, "
            f"Final test RMSE: {final_test_metrics['rmse']:.4f}"
        )

        if final_gate is not None:
            print(
                f"Average MoE gate weights on test set "
                f"[MolSim, ActCliff]: {np.round(final_gate, 4)}"
            )

    results_df = pd.DataFrame(all_run_data)

    csv_file_path = os.path.join(
        result_dir,
        'summary.csv'
    )

    results_df.to_csv(
        csv_file_path,
        index=False
    )

    print(
        f"\nAll results and hyperparameters saved to: {csv_file_path}"
    )


if __name__ == '__main__':
    config_path = (
        'finetune.yaml'
    )

    with open(config_path, 'r') as f:
        config = yaml.load(
            f,
            Loader=yaml.FullLoader
        )

    config['num_run'] = 5

    config['dataset']['data_dir'] = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..',
            'data',
            'finetune',
        )
    )

    config['device'] = (
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    config['model']['name'] = 'gat'

    if 'return_attention_weights' not in config['model']:
        config['model']['return_attention_weights'] = True

    print("\nCurrent configuration:")
    print("\nCurrent configuration:")
    print(
        yaml.dump(
            config,
            default_flow_style=False,
            sort_keys=False
        )
    )

    main(config)