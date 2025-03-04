import torch.nn as nn
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention

from template.config import Config


def set_hyperparameters(
    cfg: Config, max_epochs: int, hidden_dim: int, lr: float, weight_decay: float
) -> Config:
    cfg.trainer.max_epochs = max_epochs
    cfg.model.hidden_dim = 2**hidden_dim
    cfg.optimizer.lr = lr
    cfg.optimizer.weight_decay = weight_decay
    return cfg


def positional_encoding(score, batch, head, q_idx, kv_idx):
    """Relative positional encoding"""
    return score + (q_idx - kv_idx)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.pre_norm = nn.RMSNorm(normalized_shape=embedding_dim)
        self.post_norm = nn.RMSNorm(normalized_shape=embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=hidden_dim),
            nn.SiLU(),
            nn.Linear(in_features=hidden_dim, out_features=embedding_dim),
        )
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

    def multi_head_attention(self, x: Tensor) -> Tensor:
        # (B, L, E) -> (B, H, L, D)
        x = x.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)
        score = flex_attention(query=x, key=x, value=x, score_mod=positional_encoding)
        # (B, H, L, D) -> (B, L, E)
        score = score.transpose(1, 2).flatten(-2)
        return score

    def forward(self, x: Tensor):
        x = self.pre_norm(x)
        x = x + self.multi_head_attention(x)
        x = self.post_norm(x)
        x = x + self.feed_forward(x)
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

    def forward(self, x: Tensor):
        x = self.embeddings(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = x.mean(dim=1)
        return self.mlp(x)
