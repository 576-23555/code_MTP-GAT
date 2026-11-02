import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import LayerNorm


class GAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers, num_heads,
                 final_out_dim, dropout, alpha=0.2, use_norm=False, norm_type='layernorm'):
        super(GAT, self).__init__()
        self.name = 'gat'
        self.use_norm = use_norm
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_p = dropout

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.layers.append(
            GATConv(in_feats, hidden_feats, heads=num_heads, dropout=dropout, concat=True, negative_slope=alpha)
        )
        self.norms.append(LayerNorm(hidden_feats * num_heads))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                GATConv(hidden_feats * num_heads, hidden_feats, heads=num_heads, dropout=dropout, concat=True, negative_slope=alpha)
            )
            self.norms.append(LayerNorm(hidden_feats * num_heads))

        # Final layer
        if num_layers > 1:
            self.layers.append(
                GATConv(hidden_feats * num_heads, out_feats, heads=num_heads, dropout=dropout, concat=True, negative_slope=alpha)
            )
            self.norms.append(LayerNorm(out_feats * num_heads))
            self.final_dim = out_feats * num_heads
        else:
            self.final_dim = hidden_feats * num_heads

    def forward(self, data, return_attention_weights=False, return_node_embeddings=False):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        all_attn_weights = []

        for i, layer in enumerate(self.layers):
            if return_attention_weights:
                # GATConv returns (x, (edge_index, attention_weights))
                x_layer, (edge_index_layer, attn_weights_layer) = layer(x, edge_index, return_attention_weights=True)
                all_attn_weights.append((edge_index_layer, attn_weights_layer))
                x = x_layer
            else:
                x = layer(x, edge_index)

            if self.use_norm:
                x = self.norms[i](x)
            if i < len(self.layers) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout_p, training=self.training)

        if return_node_embeddings and return_attention_weights:
            return x, all_attn_weights
        if return_node_embeddings:
            return x

        return x
