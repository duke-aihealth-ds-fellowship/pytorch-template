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
from torch.utils.data import Dataset

from template.dataset import collate_fn


def make_attributions(
    input_dataset: Dataset,
    baseline_dataset: Dataset,
    model: nn.Module,
    output_idx: int,
    device: str,
):
    inputs = [instance for instance in input_dataset]
    baselines = [instance for instance in baseline_dataset]
    inputs, _ = collate_fn(inputs)
    baselines, _ = collate_fn(baselines)
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


def plot_attributions(df: pl.DataFrame):
    sns.set_theme(style="darkgrid", font_scale=1.5)
    top_k_tokens = get_top_k_tokens(df, k=20)
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
    plt.grid(axis="y")
    ax.axes.axvline(0, color="black", linestyle="--")
    norm = plt.Normalize(df["count"].min(), df["count"].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("count")
