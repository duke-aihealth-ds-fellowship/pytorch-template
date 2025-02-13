import torch
import torch.nn as nn

from template.config import Config


def set_hyperparameters(
    cfg: Config, max_epochs: int, hidden_dim: int, lr: float, weight_decay: float
) -> Config:
    cfg.trainer.max_epochs = max_epochs
    cfg.model.hidden_dim = 2**hidden_dim
    cfg.optimizer.lr = lr
    cfg.optimizer.weight_decay = weight_decay
    return cfg


class EmbeddingModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.mlp = nn.Sequential(
            nn.RMSNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=hidden_dim),
            nn.SiLU(),
            nn.Linear(in_features=hidden_dim, out_features=output_dim),
        )

    def forward(self, x: torch.Tensor):
        x = self.embeddings(x)
        x = x.mean(dim=1)
        return self.mlp(x)
