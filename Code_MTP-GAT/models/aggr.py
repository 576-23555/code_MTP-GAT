
import torch
import torch.nn as nn
import torch.nn.functional as F

class PromptAggr(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.0, local_loss=False, layer_norm_out=True):
        super(PromptAggr, self).__init__()

        assert emb_dim % num_heads == 0
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.local_loss = local_loss

        self.attn = nn.MultiheadAttention(
            emb_dim,
            num_heads,
            batch_first=True,
            dropout=dropout,
        )

        self.prompt_query = nn.Parameter(torch.FloatTensor(emb_dim))
        nn.init.normal_(self.prompt_query)

        self.linear = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        self.layer_norm_1 = nn.LayerNorm(emb_dim) if layer_norm_out else None

        if local_loss:
            self.layer_norm_2 = nn.LayerNorm(emb_dim) if layer_norm_out else None

    def forward(self, batch_x, batch_mask, return_attention_weights=False):
        batch_size = batch_x.size(0)
        batch_x = self.linear(batch_x)

        global_out, score = self.attn(
            self.prompt_query.expand(batch_size, 1, self.emb_dim),
            batch_x,
            batch_x,
            key_padding_mask=~batch_mask,
            average_attn_weights=False
        )

        global_out = global_out.squeeze(1)

        if self.layer_norm_1 is not None:
            global_out = self.layer_norm_1(global_out)

        local_out = None

        if self.local_loss:
            if self.layer_norm_2 is not None:
                local_out = self.layer_norm_2(batch_x)
            else:
                local_out = batch_x

        if return_attention_weights:
            final_attention_score = score.mean(dim=1)
            return global_out, local_out, final_attention_score
        else:
            return global_out, local_out, None


class MoEGate(nn.Module):
    def __init__(
        self,
        emb_dim,
        num_experts=2,
        hidden_dim=None,
        dropout=0.0,
        temperature=1.0,
        init_logits=None,
        zero_init=True
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.num_experts = num_experts
        self.temperature = max(float(temperature), 1e-8)
        hidden_dim = hidden_dim or emb_dim

        self.net = nn.Sequential(
            nn.Linear(num_experts * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )

        if zero_init:
            self._make_constant(gate_logits(init_logits, num_experts))

    def _make_constant(self, logits):
        last = self.net[-1]

        with torch.no_grad():
            nn.init.zeros_(last.weight)
            last.bias.copy_(logits.reshape(-1))

    def set_logits(self, logits):
        self._make_constant(gate_logits(logits, self.num_experts))

    def get_constant_logits(self):
        return self.net[-1].bias.detach()

    def forward(self, expert_feats):
        if torch.is_tensor(expert_feats) and expert_feats.dim() == 3:
            feats = torch.cat(
                [expert_feats[k] for k in range(expert_feats.size(0))],
                dim=-1
            )
        else:
            feats = expert_feats

        logits = self.net(feats)
        probs = F.softmax(logits / self.temperature, dim=-1)

        return logits, probs


def gate_logits(init, num_experts):
    if init is None:
        return torch.zeros(num_experts)

    if not torch.is_tensor(init):
        init = torch.as_tensor(init, dtype=torch.float32)

    init = init.reshape(-1).float()

    if init.numel() != num_experts:
        init = (
            torch.zeros(num_experts)
            if init.numel() == 0
            else torch.nn.functional.pad(
                init[:num_experts],
                (0, num_experts - init.numel())
            )
        )

    return init
