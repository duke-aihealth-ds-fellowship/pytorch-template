import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention

from template.config import Config


def set_hyperparameters(
    cfg: Config,
    max_epochs: int,
    hidden_dim: int,
    num_heads: int,
    num_layers: int,
    dropout: float,
    lr: float,
    weight_decay: float,
) -> Config:
    cfg.trainer.max_epochs = max_epochs
    cfg.model.hidden_dim = 2**hidden_dim
    cfg.model.num_heads = 2**num_heads
    cfg.model.num_layers = num_layers
    cfg.model.dropout = dropout
    cfg.optimizer.lr = lr
    cfg.optimizer.weight_decay = weight_decay
    return cfg


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        multiple_of,
        ffn_dim_multiplier=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False, **factory_kwargs)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False, **factory_kwargs)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False, **factory_kwargs)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        assert embedding_dim % num_heads == 0, (
            f"embedding_dim {embedding_dim} must be divisible by num_heads {num_heads}"
        )

    def forward(self, x: Tensor) -> Tensor:
        # (B, L, E) -> (B, H, L, D)
        x = x.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)
        score = flex_attention(
            query=x, key=x, value=x, score_mod=positional_encoding, enable_gqa=True
        )
        # (B, H, L, D) -> (B, L, E)
        score = score.transpose(1, 2).flatten(-2)
        return score


def positional_encoding(score, batch, head, q_idx, kv_idx):
    """Relative positional encoding"""
    return score + (q_idx - kv_idx)


class TransformerBlock(nn.Module):
    def __init__(
        self, embedding_dim: int, num_heads: int, hidden_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.pre_norm = nn.RMSNorm(normalized_shape=embedding_dim)
        self.feed_forward = SwiGLUFFN(
            dim=embedding_dim, hidden_dim=hidden_dim, multiple_of=64
        )
        self.post_norm = nn.RMSNorm(normalized_shape=embedding_dim)
        self.multi_head_attention = MultiHeadAttention(
            embedding_dim=embedding_dim, num_heads=num_heads
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor):
        residual = x
        x = self.pre_norm(x)
        attn_output = self.multi_head_attention(x)
        attn_output = self.dropout(attn_output)
        x = residual + attn_output
        residual = x
        x = self.post_norm(x)
        ff_output = self.feed_forward(x)
        ff_output = self.dropout(ff_output)
        x = residual + ff_output
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.mlp = nn.Sequential(
            nn.RMSNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=hidden_dim),
            nn.SiLU(),
            nn.Linear(in_features=hidden_dim, out_features=output_dim),
        )

    def masked_mean_pool(self, x: Tensor, mask: Tensor):
        # x: (B, L, E)
        # mask: (B, L)
        expanded_mask = mask.unsqueeze(-1)  # (B, L, 1)
        masked_sum = (x * expanded_mask).sum(dim=1)  # (B, E)
        token_count = expanded_mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
        x = masked_sum / token_count
        return x

    def forward(self, x: Tensor):
        padding_mask = x == 0
        x = self.embeddings(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.masked_mean_pool(x, padding_mask)
        return self.mlp(x)
