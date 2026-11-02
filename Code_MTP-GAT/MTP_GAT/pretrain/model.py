import numpy as np
from itertools import product
from rdkit import DataStructs
from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from gat.gat import GAT
from pretrain.sim import compute_all_acs_matrices_with_penalty


class PromptAggr(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout, local_loss, layer_norm_out):
        super().__init__()
        self.aggr_layer = nn.MultiheadAttention(
            emb_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.local_loss = local_loss

        self.layer_norm_out = layer_norm_out
        if self.layer_norm_out:
            self.norm_layer = nn.LayerNorm(emb_dim)

    def forward(self, fx, fm, return_attention_weights=True):
        query = self.query.expand(fx.size(0), -1, -1)

        key_padding_mask = ~fm

        attn_output, attn_weights = self.aggr_layer(
            query, fx, fx,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention_weights
        )

        output = attn_output.squeeze(1)

        if self.layer_norm_out:
            output = self.norm_layer(output)

        return output, None, attn_weights.squeeze(1)


def get_longest_alkyl_chain_length(mol):
    if mol is None:
        return 0

    carbon_graph = {}

    for atom in mol.GetAtoms():
        if (
            atom.GetAtomicNum() == 6
            and not atom.GetIsAromatic()
            and not atom.IsInRing()
            and atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3
        ):
            carbon_graph[atom.GetIdx()] = []

    for bond in mol.GetBonds():
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()

        if (
            idx1 in carbon_graph
            and idx2 in carbon_graph
            and bond.GetBondType() == Chem.rdchem.BondType.SINGLE
        ):
            carbon_graph[idx1].append(idx2)
            carbon_graph[idx2].append(idx1)

    if not carbon_graph:
        return 0

    def bfs_farthest(start, component):
        distances = {start: 0}
        queue = [start]

        for node in queue:
            for neighbor in carbon_graph[node]:
                if neighbor in component and neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)

        farthest_node = max(distances, key=distances.get)
        return farthest_node, distances[farthest_node]

    visited = set()
    max_chain_length = 0

    for start in carbon_graph:
        if start in visited:
            continue

        component = set()
        stack = [start]

        while stack:
            node = stack.pop()

            if node in component:
                continue

            component.add(node)
            visited.add(node)

            stack.extend(
                neighbor
                for neighbor in carbon_graph[node]
                if neighbor not in component
            )

        endpoint, _ = bfs_farthest(start, component)
        _, max_distance = bfs_farthest(endpoint, component)

        max_chain_length = max(max_chain_length, max_distance + 1)

    return max_chain_length

def _to_explicit_bitvect_helper(arr):
    if not isinstance(arr, np.ndarray) or arr.ndim != 1:
        return None
    if arr.dtype != np.bool_:
        arr = arr.astype(bool)

    bv = DataStructs.ExplicitBitVect(len(arr))
    on_bits = np.nonzero(arr)[0]
    for idx in on_bits:
        bv.SetBit(int(idx))
    return bv

def calculate_penalized_similarity(fp1, fp2, mol1, mol2, penalty_coeff=0.1):
    if fp1 is None or fp2 is None or mol1 is None or mol2 is None:
        return 0.0
    fp1_bv = fp1 if isinstance(fp1, DataStructs.ExplicitBitVect) else _to_explicit_bitvect_helper(fp1)
    fp2_bv = fp2 if isinstance(fp2, DataStructs.ExplicitBitVect) else _to_explicit_bitvect_helper(fp2)

    if fp1_bv is None or fp2_bv is None:
        return 0.0

    base_sim = DataStructs.FingerprintSimilarity(fp1_bv, fp2_bv)
    chain_len1 = get_longest_alkyl_chain_length(mol1)
    chain_len2 = get_longest_alkyl_chain_length(mol2)

    penalty = np.log(1 + abs(chain_len1 - chain_len2))
    adjusted_sim = base_sim * np.exp(-penalty_coeff * penalty)

    return adjusted_sim

class GNNWrapper(nn.Module):
    def __init__(self, gnn, emb_dim, config, atom_context_size, fp_dim=None,
                 total_views_per_original_mol_all_tasks=None):
        super().__init__()
        self.fp_dim = fp_dim

        print("[DEBUG] 初始化 GNNWrapper, fp_dim =", self.fp_dim) 
        self.device = config['device']
        self.margin = config['optim']['margin']
        self.adaptive_margin_coeff = config['optim']['adamg_coeff']
        self.dist_metric = config['optim']['distance_metric']
        
        self.dropout_ratio = config['model']['dropout_ratio']
        self.attention_reg_coeff = config['model'].get('attention_reg_coeff', 0.001)

        self.activity_pred_loss_weight = config['optim'].get('activity_pred_loss_weight', 0.1) 

        self.molsim_num_candidates = config['optim'].get('num_candidates', 1)
        self.molsim_num_random_negatives = config['optim'].get('num_random_negatives', 4)
        self.molsim_total_views_per_original_mol = 1 + self.molsim_num_candidates + self.molsim_num_random_negatives
        self.temperature_molsim = config['optim'].get('temperature_molsim', 0.1)

        self.actcliff_num_explicit_negatives = config['optim'].get('num_explicit_negatives', 1)
        self.actcliff_num_additional_negatives = config['optim'].get('num_additional_negatives', 2)
        self.actcliff_total_views_per_original_mol = 1 + 1 + self.actcliff_num_explicit_negatives + self.actcliff_num_additional_negatives
        self.explicit_neg_weight = config['optim'].get('explicit_neg_weight', 2.0)
        self.temperature_actcliff = config['optim'].get('temperature_actcliff', 0.1)
        self.actcliff_adaptive_margin_coeff = config['optim'].get('actcliff_adamg_coeff', 0.5)

        aggr_head = config['model'].get('aggr_head', 4)
        layer_norm_out = config['model'].get('layer_norm_out', True)

        self.aggrs = nn.ModuleList([
            PromptAggr(emb_dim=emb_dim, num_heads=aggr_head, dropout=self.dropout_ratio,
                       local_loss=False, layer_norm_out=layer_norm_out),
            PromptAggr(emb_dim=emb_dim, num_heads=aggr_head, dropout=self.dropout_ratio,
                       local_loss=False, layer_norm_out=layer_norm_out)
        ])

        self.emb_dim = emb_dim 

        feat_type = config.get('feat_type', 'rich')
        if feat_type == 'rich':
            atom_feat_dim = 143
            bond_feat_dim = 14
        else:
            atom_feat_dim = config.get('atom_feat_dim', 143)  # 你可以改成合适的默认值
            bond_feat_dim = config.get('bond_feat_dim', 14)

        gnn_name = config['model']['name'].lower()
        gnn_params = config['model'][f'gnn_params_{gnn_name}']

        _input_dim = gnn_params.get('in_feats') or gnn_params.get('node_dim')
        _dropout = config['model'].get('dropout_ratio', 0.3)
        _final_out_dim = gnn_params.get('final_out_dim', 1)
        _expected_emb_dim = self.emb_dim  # 期望的embedding维度

        gnn_name = config['model']['name'].lower() 

        gnn_params = config['model'][f'gnn_params_{gnn_name}']

        if 'in_feats' not in gnn_params and 'node_dim' not in gnn_params:
            raise ValueError("GAT configuration must contain the 'in_feats' parameter.")
        
        _input_dim = gnn_params.get('in_feats') or gnn_params.get('node_dim')
        _dropout = self.dropout_ratio
        _final_out_dim = gnn_params.get('final_out_dim', 1)

        if gnn_name != 'gat':
            raise ValueError(
                f"This GNNWrapper supports only GAT, but received: {gnn_name}."
            )

        self.gnn = GAT(
            in_feats=_input_dim,
            hidden_feats=gnn_params.get('hidden_feats'),
            out_feats=gnn_params.get('out_feats'),
            num_layers=gnn_params.get('num_layers'),
            num_heads=gnn_params.get('num_heads'),
            dropout=_dropout,
            alpha=gnn_params.get('alpha'),
            final_out_dim=_final_out_dim,
            use_norm=gnn_params.get('use_norm', False),
            norm_type=gnn_params.get('norm_type', 'layernorm')
        )

        final_node_emb_dim = gnn_params.get('out_feats') * gnn_params.get('num_heads')
        if final_node_emb_dim != self.emb_dim:
            raise ValueError(
                f"GAT model output dimension ({final_node_emb_dim}) "
                f"does not match PromptAggr embedding dimension ({self.emb_dim}). "
                "Please adjust out_feats or num_heads."
            )

        self.criterion = nn.CrossEntropyLoss(reduction="mean")

        if self.fp_dim is not None and self.fp_dim > 0:
            hidden_fp_dim = 512 
            self.fp_projection = nn.Sequential(
                nn.Linear(self.emb_dim, hidden_fp_dim),
                nn.ReLU(),
                nn.Linear(hidden_fp_dim, self.fp_dim)
            )
        else:
            self.fp_projection = None

        self.activity_prediction_head = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim // 2),
            nn.ReLU(),
            nn.Linear(self.emb_dim // 2, 1)
        )

        if total_views_per_original_mol_all_tasks is None:
            self.total_views_per_original_mol_all_tasks = \
                self.molsim_total_views_per_original_mol + self.actcliff_total_views_per_original_mol
        else:
            self.total_views_per_original_mol_all_tasks = total_views_per_original_mol_all_tasks

    def _to_explicit_bitvect(self, arr):
        return _to_explicit_bitvect_helper(arr)

    def distance(self, tensor_a, tensor_b, metric=None):
        chosen_metric = metric if metric is not None else self.dist_metric
        if chosen_metric == 'cossim':
            return 1 - F.cosine_similarity(tensor_a, tensor_b, dim=-1)
        elif chosen_metric == 'l2norm':
            return (tensor_a - tensor_b).norm(dim=-1)
        else:
            raise ValueError(f"Unsupported distance metric: {chosen_metric}")

    def forward(self, batch, task_type):
        x = batch.x.float()
        gnn_name = self.gnn.name 
        if gnn_name != 'gat':
            raise ValueError(f"Expected GAT backbone, got: {gnn_name}")

        h_node = self.gnn(batch, return_node_embeddings=True)

        batch_x, batch_mask = to_dense_batch(h_node, batch.batch)

        if task_type == 'molsim':
            print("batch_x.shape:", batch_x.shape)
            global_out_molsim, _, aggr_score_molsim = self.aggrs[0](batch_x, batch_mask, return_attention_weights=True)
            if aggr_score_molsim is None:
                raise RuntimeError("MolSim aggr_score from PromptAggr is None. Check PromptAggr implementation.")
            return self._compute_molsim_loss(global_out_molsim, aggr_score_molsim, batch, batch_mask) # 传入 batch_mask

        elif task_type == 'actcliff':
            global_out_actcliff, _, aggr_score_actcliff = self.aggrs[1](batch_x, batch_mask, return_attention_weights=True)
            if aggr_score_actcliff is None:
                raise RuntimeError("ActCliff aggr_score from PromptAggr is None. Check PromptAggr implementation.")
            return self._compute_actcliff_loss(global_out_actcliff, aggr_score_actcliff, batch, batch_mask) # 传入 batch_mask
        else:
            raise ValueError(f"未知的任务类型: {task_type}. 必须是 'molsim' 或 'actcliff'.")

    def _compute_molsim_loss(self, h_gr_molsim, aggr_score_molsim, batch, batch_mask): # 接收 batch_mask
        num_graphs_in_batch = len(batch.batch.unique())


        if num_graphs_in_batch % self.molsim_total_views_per_original_mol != 0:
            raise ValueError(
                f"MolSim 批次中的图数量 ({num_graphs_in_batch}) "
                f"不是每个原始分子视图数量 ({self.molsim_total_views_per_original_mol}) 的整数倍。 "
                "请检查数据加载器（例如设置 drop_last=True）或数据处理逻辑。"
            )

        batch_size_original_mols = num_graphs_in_batch // self.molsim_total_views_per_original_mol

        if batch_size_original_mols == 0:
            return {'loss_1': torch.tensor(0.0, device=self.device),
                    'loss_1_reg': torch.tensor(0.0, device=self.device),
                    'loss_2': torch.tensor(0.0, device=self.device),
                    'loss_2_reg': torch.tensor(0.0, device=self.device)}

        smi_for_molsim = [batch.smi[i] for i in range(0, num_graphs_in_batch, self.molsim_total_views_per_original_mol)]
        
        mol_fps_np_for_molsim = [batch.mol_fps[i].cpu() for i in range(0, num_graphs_in_batch, self.molsim_total_views_per_original_mol)]

        mol_objs_for_molsim = [Chem.MolFromSmiles(s) for s in smi_for_molsim]

        validated_fps_molsim = []
        for fp_arr in mol_fps_np_for_molsim:
            try:
                if not isinstance(fp_arr, np.ndarray):
                    fp_arr = np.array(fp_arr)
                converted_fp = self._to_explicit_bitvect(fp_arr)
                if converted_fp is None:
                    print(f"Warning: Failed to convert fingerprint for array of shape {fp_arr.shape} and type {fp_arr.dtype}. Appending None.")
                validated_fps_molsim.append(converted_fp)
            except Exception as e:
                print(f"Warning: Error converting fingerprint: {e}. Skipping this fingerprint.")
                validated_fps_molsim.append(None)

        penalized_sim_matrix_molsim = torch.zeros((batch_size_original_mols, batch_size_original_mols), device=self.device)
        for i, j in product(range(batch_size_original_mols), range(batch_size_original_mols)):
            if validated_fps_molsim[i] is not None and validated_fps_molsim[j] is not None and \
            mol_objs_for_molsim[i] is not None and mol_objs_for_molsim[j] is not None:
                penalized_sim_matrix_molsim[i, j] = calculate_penalized_similarity(
                    validated_fps_molsim[i], validated_fps_molsim[j], mol_objs_for_molsim[i], mol_objs_for_molsim[j]
                )
            else:
                penalized_sim_matrix_molsim = torch.zeros_like(penalized_sim_matrix_molsim) # 确保维度一致
                

        mol_dist_molsim = (1 - penalized_sim_matrix_molsim).to(self.device)

        h_gr_molsim_reshaped = h_gr_molsim.view(batch_size_original_mols, self.molsim_total_views_per_original_mol, self.emb_dim)
        h_gr_molsim_normalized = F.normalize(h_gr_molsim_reshaped, dim=-1)

        loss_1 = self.adaptive_margin_loss(h_gr_molsim_normalized, mol_dist_molsim)

        num_non_zeros_per_graph = (aggr_score_molsim * batch_mask).gt(0).sum(dim=-1, keepdim=True).float()
        target_scale_per_graph = torch.where(num_non_zeros_per_graph > 0, 1.0 / num_non_zeros_per_graph, torch.tensor(0.0).to(self.device))
        target_aggr_score_molsim = torch.zeros_like(aggr_score_molsim).to(self.device)
        target_aggr_score_for_loss = torch.zeros_like(aggr_score_molsim)

        broadcasted_target_scale = target_scale_per_graph.expand_as(aggr_score_molsim)
        
        target_aggr_score_for_loss = torch.where(
            (aggr_score_molsim > 0) & batch_mask,
            broadcasted_target_scale,
            torch.tensor(0.0).to(self.device)
        )
        
        loss_1_reg = self.attention_reg_coeff * F.smooth_l1_loss(
            aggr_score_molsim[batch_mask],
            target_aggr_score_for_loss[batch_mask],
            reduction='mean'
        )

        if self.fp_projection is not None:
            actual_anchor_fps_np = [batch.mol_fps[i].cpu() for i in range(0, num_graphs_in_batch, self.molsim_total_views_per_original_mol)]
            actual_anchor_fps_tensor = torch.stack(actual_anchor_fps_np).to(self.device)
            
            anchor_mol_emb = h_gr_molsim_reshaped[:, 0, :]
            predicted_fp_molsim = self.fp_projection(anchor_mol_emb)
            predicted_fp_molsim = torch.sigmoid(predicted_fp_molsim)
            loss_1_fp_pred = F.binary_cross_entropy(predicted_fp_molsim, actual_anchor_fps_tensor)
            loss_1_reg = loss_1_reg + loss_1_fp_pred
        else:
            loss_1_fp_pred = torch.tensor(0.0, device=self.device)

        return {
            'loss_1': loss_1,
            'loss_1_reg': loss_1_reg,
            'loss_2': torch.tensor(0.0, device=self.device),
            'loss_2_reg': torch.tensor(0.0, device=self.device)
        }

    def _compute_actcliff_loss(self, h_gr_actcliff, aggr_score_actcliff, batch, batch_mask): # 接收 batch_mask
        num_graphs_in_batch = len(batch.batch.unique())

        if num_graphs_in_batch % self.actcliff_total_views_per_original_mol != 0:
            raise ValueError(
                f"ActCliff 批次中的图数量 ({num_graphs_in_batch}) "
                f"不是每个原始分子视图数量 ({self.actcliff_total_views_per_original_mol}) 的整数倍。 "
                "请检查数据加载器（例如设置 drop_last=True）或数据处理逻辑。"
            )

        batch_size_original_mols = num_graphs_in_batch // self.actcliff_total_views_per_original_mol

        if batch_size_original_mols == 0:
            return {'loss_1': torch.tensor(0.0, device=self.device),
                                           'loss_1_reg': torch.tensor(0.0, device=self.device),
                                           'loss_2': torch.tensor(0.0, device=self.device),
                                           'loss_2_reg': torch.tensor(0.0, device=self.device)}

        if not hasattr(batch, 'mol_md') or batch.mol_md is None or batch.mol_md.numel() == 0:
            print("Warning: Batch object does not contain valid 'mol_md' attribute for Activity Cliff calculation. Returning zero loss.")
            return {'loss_1': torch.tensor(0.0, device=self.device),
                                           'loss_1_reg': torch.tensor(0.0, device=self.device),
                                           'loss_2': torch.tensor(0.0, device=self.device),
                                           'loss_2_reg': torch.tensor(0.0, device=self.device)}
        
        if batch.mol_md.ndim > 1 and batch.mol_md.shape[1] == 1:
            batch.mol_md = batch.mol_md.squeeze(1)
        elif batch.mol_md.ndim > 1:
            raise ValueError(f"batch.mol_md has unexpected shape {batch.mol_md.shape}. Expected (N,) or (N,1) for _compute_actcliff_loss. If it's multi-dimensional (N, D), you might need to flatten or aggregate D.")
        
        expected_numel = batch_size_original_mols * self.actcliff_total_views_per_original_mol
        
        if batch.mol_md.numel() != expected_numel:
            raise RuntimeError(
                f"内部错误: batch.mol_md 的实际元素数量 ({batch.mol_md.numel()}) "
                f"与期望的重塑形状 ({batch_size_original_mols}, {self.actcliff_total_views_per_original_mol}) "
                f"的元素数量 ({expected_numel}) 不匹配。这表明数据批处理存在深层问题。"
            )

        mds_reshaped = batch.mol_md.view(batch_size_original_mols, self.actcliff_total_views_per_original_mol)

        h_gr_actcliff_reshaped = h_gr_actcliff.view(batch_size_original_mols, self.actcliff_total_views_per_original_mol, self.emb_dim)
        h_gr_actcliff_normalized = F.normalize(h_gr_actcliff_reshaped, dim=-1)

        original_mol_smis = [batch.smi[i * self.actcliff_total_views_per_original_mol] for i in range(batch_size_original_mols)]
        original_mols = [Chem.MolFromSmiles(s) for s in original_mol_smis]

        precomputed_acs_matrices = compute_all_acs_matrices_with_penalty(original_mols)

        loss_2 = self._compute_actcliff_adaptive_loss(
            h_gr_actcliff_normalized,
            mds_reshaped,
            batch_size_original_mols,
            precomputed_acs_matrices,
            batch
        )

        num_non_zeros_per_graph = (aggr_score_actcliff * batch_mask).gt(0).sum(dim=-1, keepdim=True).float()
        target_scale_per_graph = torch.where(num_non_zeros_per_graph > 0, 1.0 / num_non_zeros_per_graph, torch.tensor(0.0).to(self.device))

        target_aggr_score_for_loss = torch.zeros_like(aggr_score_actcliff)
        broadcasted_target_scale = target_scale_per_graph.expand_as(aggr_score_actcliff)
        
        target_aggr_score_for_loss = torch.where(
            (aggr_score_actcliff > 0) & batch_mask,
            broadcasted_target_scale,
            torch.tensor(0.0).to(self.device)
        )

        loss_2_reg = self.attention_reg_coeff * F.smooth_l1_loss(
            aggr_score_actcliff[batch_mask],
            target_aggr_score_for_loss[batch_mask],
            reduction='mean'
        )


        all_graph_embeddings = h_gr_actcliff 
        predicted_activities = self.activity_prediction_head(all_graph_embeddings).squeeze(-1)

        true_activities = batch.mol_md.float()

        if predicted_activities.shape != true_activities.shape:
            raise ValueError(f"预测活性形状 {predicted_activities.shape} 与真实活性形状 {true_activities.shape} 不匹配。")

        loss_activity_pred = F.mse_loss(predicted_activities, true_activities, reduction='mean')
        
        loss_2_reg = loss_2_reg + self.activity_pred_loss_weight * loss_activity_pred

        return {
            'loss_1': torch.tensor(0.0, device=self.device),
            'loss_1_reg': torch.tensor(0.0, device=self.device),
            'loss_2': loss_2,
            'loss_2_reg': loss_2_reg
        }

    def _compute_actcliff_adaptive_loss(self, embeddings, activity_values, batch_size_original_mols, acs_matrices, batch_data):
        if batch_size_original_mols == 0:
            return torch.tensor(0.0, device=self.device)

        anchor_reps = embeddings[:, 0, :]
        similar_reps = embeddings[:, 1, :]
        all_local_neg_reps = embeddings[:, 2:, :]

        anchor_mds = activity_values[:, 0]
        similar_mds = activity_values[:, 1]
        all_local_neg_mds = activity_values[:, 2:]

        d_ap = self.distance(anchor_reps, similar_reps)

        explicit_neg_reps = all_local_neg_reps[:, :self.actcliff_num_explicit_negatives, :]
        additional_neg_reps = all_local_neg_reps[:, self.actcliff_num_explicit_negatives:, :]

        d_an_explicit = self.distance(anchor_reps.unsqueeze(1), explicit_neg_reps) if explicit_neg_reps.size(1) > 0 else torch.empty((batch_size_original_mols, 0), device=self.device)
        d_an_additional = self.distance(anchor_reps.unsqueeze(1), additional_neg_reps) if additional_neg_reps.size(1) > 0 else torch.empty((batch_size_original_mols, 0), device=self.device)

        inter_anchor_dists = self.distance(anchor_reps.unsqueeze(1), anchor_reps.unsqueeze(0))
        diag_idx = torch.eye(batch_size_original_mols, dtype=torch.bool, device=self.device)
        d_an_inter_batch = inter_anchor_dists[~diag_idx].view(batch_size_original_mols, batch_size_original_mols - 1)

        d_an_list = []
        if d_an_explicit.size(1) > 0:
            d_an_list.append(d_an_explicit)
        if d_an_additional.size(1) > 0:
            d_an_list.append(d_an_additional)
        if d_an_inter_batch.size(1) > 0:
            d_an_list.append(d_an_inter_batch)

        if not d_an_list:
            return torch.tensor(0.0, device=self.device)

        d_an = torch.cat(d_an_list, dim=1)

        activity_diff_ap = torch.abs(anchor_mds - similar_mds)

        explicit_neg_mds = all_local_neg_mds[:, :self.actcliff_num_explicit_negatives]
        additional_neg_mds = all_local_neg_mds[:, self.actcliff_num_explicit_negatives:]

        activity_diff_an_explicit = torch.abs(anchor_mds.unsqueeze(1) - explicit_neg_mds) if explicit_neg_mds.numel() > 0 else torch.empty((batch_size_original_mols, 0), device=self.device)
        activity_diff_an_additional = torch.abs(anchor_mds.unsqueeze(1) - additional_neg_mds) if additional_neg_mds.numel() > 0 else torch.empty((batch_size_original_mols, 0), device=self.device)

        anchor_mds_expanded_for_inter = anchor_mds.unsqueeze(1).expand(-1, batch_size_original_mols)
        inter_anchor_mds = anchor_mds_expanded_for_inter[~diag_idx].view(batch_size_original_mols, batch_size_original_mols - 1)
        activity_diff_an_inter_batch = torch.abs(anchor_mds.unsqueeze(1) - inter_anchor_mds)

        activity_diff_an_list = []
        if activity_diff_an_explicit.size(1) > 0:
            activity_diff_an_list.append(activity_diff_an_explicit)
        if activity_diff_an_additional.size(1) > 0:
            activity_diff_an_list.append(activity_diff_an_additional)
        if activity_diff_an_inter_batch.size(1) > 0:
            activity_diff_an_list.append(activity_diff_an_inter_batch)

        if not activity_diff_an_list:
            return torch.tensor(0.0, device=self.device)

        activity_diff_an = torch.cat(activity_diff_an_list, dim=1)

        d_ap_expanded = d_ap.unsqueeze(1).expand_as(d_an)

        neg_weights_matrix = torch.ones_like(d_an, dtype=torch.float, device=self.device)
        if self.actcliff_num_explicit_negatives > 0 and d_an_explicit.size(1) > 0:
            neg_weights_matrix[:, :self.actcliff_num_explicit_negatives] = self.explicit_neg_weight

        current_margin_actcliff = self.margin + self.actcliff_adaptive_margin_coeff * activity_diff_an

        anchor_smi_indices_in_batch = []
        explicit_neg_smi_indices_in_batch = []

        for i in range(batch_size_original_mols):
            start_idx = i * self.actcliff_total_views_per_original_mol
            anchor_smi_indices_in_batch.append(start_idx)
            for j in range(self.actcliff_num_explicit_negatives):
                explicit_neg_smi_indices_in_batch.append(start_idx + 2 + j)

        max_smi_idx_available = len(batch_data.smi) - 1
        if anchor_smi_indices_in_batch and max(anchor_smi_indices_in_batch) > max_smi_idx_available:
            raise IndexError(f"anchor_smi_indices_in_batch 索引超出范围。最大: {max_smi_idx_available}, 实际: {max(anchor_smi_indices_in_batch)}.")
        if explicit_neg_smi_indices_in_batch and max(explicit_neg_smi_indices_in_batch) > max_smi_idx_available:
            raise IndexError(f"explicit_neg_smi_indices_in_batch 索引超出范围。最大: {max_smi_idx_available}, 实际: {max(explicit_neg_smi_indices_in_batch)}.")

        anchor_smis = [batch_data.smi[idx] for idx in anchor_smi_indices_in_batch]
        explicit_neg_smis = [batch_data.smi[idx] for idx in explicit_neg_smi_indices_in_batch]

        anchor_mols = [Chem.MolFromSmiles(s) for s in anchor_smis]
        explicit_neg_mols = [Chem.MolFromSmiles(s) for s in explicit_neg_smis]

        anchor_fps_np = [batch_data.mol_fps[idx][0] for idx in anchor_smi_indices_in_batch]
        explicit_neg_fps_np = [batch_data.mol_fps[idx][0] for idx in explicit_neg_smi_indices_in_batch]

        validated_anchor_fps_bv = [self._to_explicit_bitvect(fp_arr) for fp_arr in anchor_fps_np]
        validated_explicit_neg_fps_bv = [self._to_explicit_bitvect(fp_arr) for fp_arr in explicit_neg_fps_np]

        explicit_neg_sim_matrix = torch.zeros((batch_size_original_mols, self.actcliff_num_explicit_negatives), device=self.device)

        for i in range(batch_size_original_mols):
            for j in range(self.actcliff_num_explicit_negatives):
                flat_neg_idx = i * self.actcliff_num_explicit_negatives + j
                if validated_anchor_fps_bv[i] is not None and validated_explicit_neg_fps_bv[flat_neg_idx] is not None and \
                anchor_mols[i] is not None and explicit_neg_mols[flat_neg_idx] is not None:
                    explicit_neg_sim_matrix[i, j] = calculate_penalized_similarity(
                        validated_anchor_fps_bv[i], validated_explicit_neg_fps_bv[flat_neg_idx],
                        anchor_mols[i], explicit_neg_mols[flat_neg_idx]
                    )
                else:
                    explicit_neg_sim_matrix[i, j] = 0.0

        alpha_gate = 12.0
        tau_gate = 0.6
        gate_weights_explicit = torch.sigmoid(alpha_gate * (explicit_neg_sim_matrix - tau_gate))

        full_gate_weights = torch.ones_like(d_an, dtype=torch.float, device=self.device)
        if self.actcliff_num_explicit_negatives > 0:
            full_gate_weights[:, :self.actcliff_num_explicit_negatives] = gate_weights_explicit
        gate_weights = full_gate_weights

        loss_matrix = F.relu(d_ap_expanded - d_an + current_margin_actcliff)
        weighted_loss_matrix = loss_matrix * neg_weights_matrix * gate_weights
        final_loss = weighted_loss_matrix.mean()

        return final_loss


    def adaptive_margin_loss(self, graph_rep, mol_dist):
        batch_size_ori = graph_rep.size(0)
        num_molsim_views = graph_rep.size(1)

        anchor_reps = graph_rep[:, 0, :].unsqueeze(1)
        pos_candidates_reps = graph_rep[:, 1:, :]

        pos_dists = self.distance(anchor_reps, pos_candidates_reps)

        ori_anchor_reps = graph_rep[:, 0, :]
        sim_matrix_inter_anchor = self.distance(ori_anchor_reps.unsqueeze(1), ori_anchor_reps.unsqueeze(0))

        diag_idx = torch.eye(batch_size_ori, dtype=torch.bool, device=self.device)
        neg_dists_inter_anchor = sim_matrix_inter_anchor[~diag_idx].view(batch_size_ori, batch_size_ori - 1)

        pos_dists_expanded = pos_dists.unsqueeze(2).expand(-1, -1, batch_size_ori - 1)
        neg_dists_expanded = neg_dists_inter_anchor.unsqueeze(1).expand(-1, num_molsim_views - 1, -1)

        margin_factor_for_neg = mol_dist[~diag_idx].view(batch_size_ori, batch_size_ori - 1)
        margin_factor_for_neg_expanded = margin_factor_for_neg.unsqueeze(1).expand(-1, num_molsim_views - 1, -1)

        current_margin = self.margin + self.adaptive_margin_coeff * margin_factor_for_neg_expanded

        pairwise_margin_loss_part = torch.maximum(torch.tensor(0.0).to(self.device),
                                                 current_margin + pos_dists_expanded - neg_dists_expanded).mean()

        mol_dist_diff = mol_dist.unsqueeze(1) - mol_dist.unsqueeze(2)
        inds_triplet = torch.argwhere(mol_dist_diff > 0.3)

        inds_triplet = inds_triplet[
            (inds_triplet[:, 0] != inds_triplet[:, 1]) &
            (inds_triplet[:, 0] != inds_triplet[:, 2]) &
            (inds_triplet[:, 1] != inds_triplet[:, 2])
        ]

        diff_values = mol_dist_diff[inds_triplet[:, 0], inds_triplet[:, 1], inds_triplet[:, 2]]

        loss_adaptive_triplet_part = torch.tensor(0.0, device=self.device)
        if len(diff_values) > 0:
            if len(diff_values) > batch_size_ori ** 2:
                sub_inds = torch.argsort(diff_values, descending=True)[:batch_size_ori ** 2]
                diff_values = diff_values[sub_inds]
                inds_triplet = inds_triplet[sub_inds]

            anchor_inds_triplet, pos_mol_inds_triplet, neg_mol_inds_triplet = \
                inds_triplet[:, 0], inds_triplet[:, 1], inds_triplet[:, 2]

            h_gr_original_mols = graph_rep[:, 0]

            anchor_rep_triplet = h_gr_original_mols[anchor_inds_triplet]
            pos_rep_triplet = h_gr_original_mols[pos_mol_inds_triplet]
            neg_rep_triplet = h_gr_original_mols[neg_mol_inds_triplet]

            d_ap_triplet = self.distance(anchor_rep_triplet, pos_rep_triplet)
            d_an_triplet = self.distance(anchor_rep_triplet, neg_rep_triplet)

            loss_adaptive_triplet_part = torch.maximum(torch.tensor(0.0).to(self.device),
                                                       d_ap_triplet - d_an_triplet + diff_values.detach() * self.margin).mean()

        total_molsim_loss = self.adaptive_margin_coeff * loss_adaptive_triplet_part + pairwise_margin_loss_part

        return total_molsim_loss