import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        multiple_of,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.w3 = nn.Linear(input_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class InputProjection(nn.Module):
    def __init__(self, input_dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        self.query_projection = nn.Linear(input_dim, input_dim, bias=bias)
        self.key_projection = nn.Linear(input_dim, input_dim, bias=bias)
        self.value_projection = nn.Linear(input_dim, input_dim, bias=bias)
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)
        # (B, L, E) -> (B, L, H, D)
        query = query.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)
        key = key.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)
        value = value.unflatten(-1, [self.num_heads, self.head_dim]).transpose(1, 2)
        return query, key, value


def relative_positional_bias(score, batch, head, q_idx, kv_idx):
    return score + (q_idx - kv_idx)


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def make_attention_mask(x: Tensor) -> BlockMask | None:
    device = str(x.device)
    block_mask = create_block_mask(
        causal_mask,
        B=None,
        H=None,
        Q_LEN=x.size(1),
        KV_LEN=x.size(1),
        device=device,
        _compile="cuda" in device,
    )
    return block_mask


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int, num_heads: int, bias: bool) -> None:
        super().__init__()

        self.input_projection = InputProjection(
            input_dim=input_dim, num_heads=num_heads, bias=bias
        )
        assert input_dim % num_heads == 0, (
            f"input_dim {input_dim} must be divisible by num_heads {num_heads}"
        )

    def forward(self, x: Tensor, mask: BlockMask | None) -> Tensor:
        query, key, value = self.input_projection(x)
        score = flex_attention(
            query=query,
            key=key,
            value=value,
            score_mod=relative_positional_bias,
            block_mask=mask,
        )
        # (B, H, L, D) -> (B, L, E)
        score = score.transpose(1, 2).flatten(-2)  # type: ignore
        return score


class TransformerBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        bias: bool,
    ) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(
            input_dim=input_dim, num_heads=num_heads, bias=bias
        )
        self.attn_norm = nn.RMSNorm(input_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.RMSNorm(input_dim)
        self.ffn = SwiGLUFFN(input_dim=input_dim, hidden_dim=hidden_dim, multiple_of=64)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: BlockMask | None = None) -> Tensor:
        attn_input = self.attn_norm(x)
        attn_output = self.mha(attn_input, mask)
        x = x + self.attn_dropout(attn_output)
        ffn_input = self.ffn_norm(x)
        ffn_output = self.ffn(ffn_input)
        x = x + self.ffn_dropout(ffn_output)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        output_dim: int,
        bias: bool,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.causal = causal
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    bias=bias,
                )
                for _ in range(num_layers)
            ]
        )
        self.prediction_head = nn.Sequential(
            nn.RMSNorm(normalized_shape=input_dim),
            nn.SiLU(),
            nn.Linear(in_features=input_dim, out_features=output_dim),
        )

    def forward(self, x: Tensor):
        attention_mask = make_attention_mask(x) if self.causal else None
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask=attention_mask)
        return self.prediction_head(x)
