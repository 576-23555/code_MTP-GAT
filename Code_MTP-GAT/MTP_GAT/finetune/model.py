
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from gat.gat import GAT
from MTP_GAT.models.aggr import PromptAggr, MoEGate


class GNNPredictor(nn.Module):

    def __init__(self,
                 emb_dim: int,
                 num_tasks: int,
                 normalize: bool = False,
                 atom_feat_dim: int = None,
                 bond_feat_dim: int = None,
                 drop_ratio: float = 0.0,
                 attn_drop_ratio: float = 0.0,
                 temperature: float = 1.0,
                 use_prompt: bool = True,
                 model_head: int = 4,
                 layer_norm_out: bool = False,
                 backbone: str = 'gat',
                 act: str = 'softmax',
                 random_init: bool = True,
                 init_weight: list = None,
                 local_loss: bool = False,
                 use_moe: bool = False,
                 moe_hidden_dim: int = None,
                 moe_dropout: float = None,
                 moe_learnable: bool = True,
                 config: dict = None):

        super(GNNPredictor, self).__init__()

        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.normalize = normalize
        self.use_prompt = use_prompt
        self.use_moe = bool(use_moe and use_prompt)
        self.temperature = temperature
        self.act = act
        self.drop_ratio = drop_ratio
        self.backbone = backbone
        self.aggr_head = config['model'].get('aggr_head', 4)
        self.local_loss = local_loss
        self.moe_learnable = moe_learnable

        self.gnn = self._create_gnn_backbone(
            atom_feat_dim=atom_feat_dim,
            bond_feat_dim=bond_feat_dim,
            drop_ratio=drop_ratio,
            attn_drop_ratio=attn_drop_ratio,
            model_head=model_head,
            layer_norm_out=layer_norm_out,
            config=config,
            emb_dim=emb_dim
        )

        self.prompt_token = ['<molsim>', '<actcliff>']

        if self.use_prompt:
            self.aggrs = nn.ModuleList([
                PromptAggr(
                    emb_dim=emb_dim,
                    num_heads=self.aggr_head,
                    dropout=self.drop_ratio,
                    layer_norm_out=layer_norm_out,
                    local_loss=self.local_loss
                )
                for _ in self.prompt_token
            ])

            if init_weight is not None:
                self.prompt_weight = nn.Parameter(torch.FloatTensor(init_weight))
            elif random_init:
                self.prompt_weight = nn.Parameter(torch.FloatTensor(len(self.prompt_token)))
                nn.init.normal_(self.prompt_weight)
            else:
                self.prompt_weight = nn.Parameter(torch.FloatTensor(len(self.prompt_token)))
                nn.init.constant_(self.prompt_weight, 1)

            if self.use_moe:
                self.moe_gate = MoEGate(
                    emb_dim=emb_dim,
                    num_experts=len(self.prompt_token),
                    hidden_dim=moe_hidden_dim,
                    dropout=self.drop_ratio if moe_dropout is None else moe_dropout,
                    temperature=self.temperature,
                    init_logits=self.prompt_weight.detach().clone(),
                )

                if not moe_learnable:
                    for p in self.moe_gate.parameters():
                        p.requires_grad = False

        self.graph_pred_linear = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_ratio),
            nn.Linear(self.emb_dim, self.emb_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.drop_ratio),
            nn.Linear(self.emb_dim // 2, self.num_tasks)
        )

    def freeze_gnn_module(self):
        for param in self.gnn.parameters():
            param.requires_grad = False

    def freeze_aggr_module(self):
        if hasattr(self, 'aggrs'):
            for aggr in self.aggrs:
                for param in aggr.parameters():
                    param.requires_grad = False

    def set_moe_trainable(self, flag=True):
        if hasattr(self, 'moe_gate'):
            for p in self.moe_gate.parameters():
                p.requires_grad = bool(flag)

    def _create_gnn_backbone(self, atom_feat_dim, bond_feat_dim,
                             drop_ratio, attn_drop_ratio, model_head,
                             layer_norm_out, config, emb_dim):

        gnn_name = self.backbone
        gnn_params = config['model'].get(f'gnn_params_{gnn_name}', {})

        _input_dim = gnn_params.get('in_feats', atom_feat_dim)
        _dropout = gnn_params.get('dropout', drop_ratio)
        _final_out_dim = gnn_params.get('final_out_dim', 1)

        if gnn_name == 'gat':
            gnn = GAT(
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

            final_node_emb_dim = (
                gnn_params.get('out_feats') *
                gnn_params.get('num_heads')
            )

            if final_node_emb_dim != self.emb_dim:
                raise ValueError(
                    f"GAT output dimension ({final_node_emb_dim}) "
                    f"does not match PromptAggr emb_dim ({self.emb_dim})"
                )
        else:
            raise NotImplementedError(
                f"GNN model {gnn_name} is not supported"
            )

        return gnn

    def get_prompt_weight(self, act='softmax'):
        if self.use_moe and hasattr(self, 'moe_gate'):
            logits = self.moe_gate.get_constant_logits()
        else:
            logits = self.prompt_weight

        if act == 'softmax':
            return F.softmax(logits / self.temperature, dim=-1)
        elif act == 'none':
            return logits
        else:
            raise NotImplementedError

    def set_prompt_weight(self, weight):
        if hasattr(self, 'prompt_weight') and isinstance(self.prompt_weight, nn.Parameter):
            with torch.no_grad():
                self.prompt_weight.copy_(
                    torch.as_tensor(
                        weight,
                        device=self.prompt_weight.device,
                        dtype=self.prompt_weight.dtype
                    )
                )

            if self.use_moe and hasattr(self, 'moe_gate'):
                self.moe_gate.set_logits(self.prompt_weight.detach())
        else:
            raise AttributeError(
                "Prompt weight parameter not found or is not an nn.Parameter."
            )

    def compute_gate_weights(self, graph_reps):
        if self.use_moe:
            feats = torch.cat(
                [graph_reps[k] for k in range(graph_reps.size(0))],
                dim=-1
            )
            _, probs = self.moe_gate(feats)
            return probs

        return self.get_prompt_weight(act=self.act).to(graph_reps.dtype)

    def _combine_experts(self, graph_reps):
        if self.use_moe:
            probs = self.compute_gate_weights(graph_reps)
            return torch.einsum(
                'kbd,bk->bd',
                graph_reps,
                probs.to(graph_reps.dtype)
            )

        weight = self.get_prompt_weight(act=self.act).to(graph_reps.dtype)

        return torch.einsum(
            'kbd,k->bd',
            graph_reps,
            weight
        )

    def get_moe_weights(self, data):
        self.eval()

        with torch.no_grad():
            node_repres = self.gnn(
                data,
                return_node_embeddings=True
            )

            batch_x, batch_mask = to_dense_batch(
                node_repres,
                data.batch
            )

            reps = []

            for aggr in self.aggrs:
                h_g, _, _ = aggr(
                    batch_x,
                    batch_mask,
                    return_attention_weights=True
                )
                reps.append(h_g)

            reps = torch.stack(reps)

            return self.compute_gate_weights(reps)

    def get_representations(self, data, channel_idx=-1,
                            return_score=False,
                            return_raw_attention=False):

        x_float = data.x.float()
        raw_attn = None

        if self.backbone == 'gat':
            if return_raw_attention:
                node_repres, raw_attn = self.gnn(
                    data,
                    return_node_embeddings=True,
                    return_attention_weights=True
                )
            else:
                node_repres = self.gnn(
                    data,
                    return_node_embeddings=True
                )
        else:
            raise NotImplementedError(
                f"{self.backbone} backbone not implemented for get_representations"
            )

        if not self.use_prompt:
            batch_x, batch_mask = to_dense_batch(
                node_repres,
                data.batch
            )

            sum_nodes = (
                batch_x * batch_mask.unsqueeze(-1)
            ).sum(dim=1)

            num_nodes = batch_mask.sum(dim=1).unsqueeze(-1)

            graph_rep_pooled = torch.where(
                num_nodes > 0,
                sum_nodes / num_nodes,
                torch.zeros_like(sum_nodes)
            )

            if self.normalize:
                graph_rep_pooled = F.normalize(
                    graph_rep_pooled,
                    dim=-1
                )

            if return_raw_attention:
                return graph_rep_pooled, None, raw_attn, None

            return graph_rep_pooled

        scores = []
        graph_reps = []
        attn_weights_list = []

        batch_x, batch_mask = to_dense_batch(
            node_repres,
            data.batch
        )

        for i in range(len(self.aggrs)):
            h_g_aggr, attn_weights_aggr, score = self.aggrs[i](
                batch_x,
                batch_mask,
                return_attention_weights=True
            )

            if self.normalize:
                h_g_aggr = F.normalize(
                    h_g_aggr,
                    dim=-1
                )

            scores.append(score)
            graph_reps.append(h_g_aggr)
            attn_weights_list.append(attn_weights_aggr)

        graph_reps = torch.stack(graph_reps)
        scores = torch.stack(scores)

        if channel_idx > -1:
            if return_score:
                if return_raw_attention:
                    return (
                        graph_reps[channel_idx],
                        scores[channel_idx],
                        raw_attn,
                        attn_weights_list
                    )

                return (
                    graph_reps[channel_idx],
                    scores[channel_idx],
                    None,
                    attn_weights_list
                )

            else:
                if return_raw_attention:
                    return (
                        graph_reps[channel_idx],
                        raw_attn,
                        attn_weights_list
                    )

                return (
                    graph_reps[channel_idx],
                    None,
                    None,
                    attn_weights_list
                )

        if return_score:
            if return_raw_attention:
                return graph_reps, scores, raw_attn, attn_weights_list

            return graph_reps, scores, None, attn_weights_list

        else:
            if return_raw_attention:
                return graph_reps, raw_attn, attn_weights_list

            return graph_reps, None, None, attn_weights_list

    def forward(self, data, channel_idx=-1,
                return_attention_weights=False,
                return_gate_weights=False):

        gate_probs = None

        if return_attention_weights:
            graph_reps, scores, raw_attn, prompt_attn = self.get_representations(
                data,
                channel_idx=channel_idx,
                return_score=True,
                return_raw_attention=True
            )

            if not self.use_prompt or channel_idx > -1:
                predict = self.graph_pred_linear(graph_reps)
            else:
                gate_probs = (
                    self.compute_gate_weights(graph_reps)
                    if self.use_moe else None
                )

                graph_rep = self._combine_experts(graph_reps)
                predict = self.graph_pred_linear(graph_rep)

            if return_gate_weights:
                return predict, scores, raw_attn, prompt_attn, gate_probs

            return predict, scores, raw_attn, prompt_attn

        if self.use_prompt:
            graph_reps, _, _, _ = self.get_representations(
                data,
                channel_idx=channel_idx,
                return_score=True
            )
        else:
            graph_reps = self.get_representations(
                data,
                channel_idx=channel_idx
            )

        if not self.use_prompt or channel_idx > -1:
            predict = self.graph_pred_linear(graph_reps)
        else:
            gate_probs = (
                self.compute_gate_weights(graph_reps)
                if self.use_moe else None
            )

            graph_rep = self._combine_experts(graph_reps)
            predict = self.graph_pred_linear(graph_rep)

        if return_gate_weights:
            return predict, graph_reps, gate_probs

        return predict, graph_reps
