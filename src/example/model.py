import torch
import torch.nn as nn


class EmbeddingModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: int,
        hidden_dim: int,
        n_layers: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.embedddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        mlp_layers = [
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
        ] * n_layers
        self.model = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=hidden_dim),
            *mlp_layers,
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=output_dim),
        )

    def forward(self, x: torch.Tensor):
        x = self.embedddings(x).mean(dim=1)
        return self.model(x)
