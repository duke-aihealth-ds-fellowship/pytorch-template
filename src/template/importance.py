import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import torch
import torch.nn as nn
from captum.attr import (
    LayerGradientShap,
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer,
)

from template.config import Config
from template.dataset import DataLoaders, collate_fn
from template.model import EmbeddingModel
from template.tune import load_best_checkpoint


def make_attributions(
    inputs: torch.Tensor,
    baselines: torch.Tensor,
    model: nn.Module,
    output_idx: int,
    device: str,
):
    inputs = inputs.to(device)
    baselines = baselines.to(device)
    model.to(device)
    interpretable_embedding = configure_interpretable_embedding_layer(
        model, "embeddings"
    )
    lig = LayerGradientShap(model, model.embeddings)
    input_embeds = interpretable_embedding.indices_to_embeddings(inputs)
    baseline_embeds = interpretable_embedding.indices_to_embeddings(baselines)
    attributions, delta = lig.attribute(
        inputs=input_embeds,
        baselines=baseline_embeds,
        return_convergence_delta=True,
        target=output_idx,
    )
    remove_interpretable_embedding_layer(model, interpretable_embedding)
    # attributions shape: (batch_size, seq_length, embedding_dim)
    print("Mean convergence delta:", delta.mean().item())
    # sum the attributions across the embedding dimension
    attributions = attributions.sum(dim=-1)
    attributions = attributions / torch.norm(attributions)
    return attributions


def format_attributions(attributions, indices, vocabulary):
    vocab_decoder = {index: word for word, index in vocabulary.items()}
    df = pl.DataFrame(
        {
            "instance": np.repeat(np.arange(indices.size(0)), indices.size(1)),
            "position": np.tile(np.arange(indices.shape[1]), indices.shape[0]),
            "index": indices.flatten().numpy(),
            "attribution": attributions.detach().flatten().numpy(),
        }
    ).with_columns(pl.col("index").replace_strict(vocab_decoder).alias("token"))
    return df


def sum_attributions(df: pl.DataFrame):
    # sum attributions and count tokens within instances
    df = (
        df.group_by("instance", "index")
        .agg(pl.col("attribution").sum(), pl.col("token").count().alias("count"))
        .sort("instance", "index")
    )
    return df


def get_top_k_tokens(df: pl.DataFrame, k: int):
    (
        df.group_by("token")
        .agg(pl.col("attribution").mean().abs())
        .sort("attribution")
        .tail(k)
    )["token"].reverse()


def plot_attributions(df: pl.DataFrame, cfg: Config):
    sns.set_theme(style="darkgrid", font_scale=1.5)
    top_k_tokens = get_top_k_tokens(df, top_k=cfg.importance.top_k)
    ax = sns.stripplot(
        data=df,
        x="attribution",
        y="token",
        hue="count",
        linestyles="",
        order=top_k_tokens,
        palette="viridis",
        legend=False,
        jitter=0.2,
        alpha=0.5,
    )
    ax.set_xlabel("SHAP value")
    ax.set_ylabel("token")
    plt.grid(axis="y")
    ax.axes.axvline(0, color="black", linestyle="--")
    norm = plt.Normalize(df["count"].min(), df["count"].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Count")
    plt.savefig(cfg.data_dir / "attributions.pdf", format="pdf")


def make_vocabulary(df: pl.DataFrame):
    unique_words = df["text"].str.split(" ").explode().unique(maintain_order=True)
    vocabulary = {word: index + 1 for index, word in enumerate(unique_words)}
    vocabulary["<PAD>"] = 0
    return vocabulary


def feature_importance(cfg: Config, loaders: DataLoaders, output_idx: int):
    model = load_best_checkpoint(cfg=cfg, model_class=EmbeddingModel)
    inputs = [instance for instance in loaders.test.dataset]
    baselines = [instance for instance in loaders.validation.dataset]
    inputs, _ = collate_fn(inputs)
    baselines, _ = collate_fn(baselines)
    attributions = make_attributions(
        input_dataset=loaders.test,
        baseline_dataset=loaders.validation,
        model=model,
        output_idx=output_idx,
        device=cfg.trainer.device,
    )
    vocabulary = make_vocabulary(loaders.train.dataset.df)
    df = format_attributions(
        attributions=attributions, indices=inputs, vocabulary=vocabulary
    )
    df = sum_attributions(df)
    plot_attributions(df)
