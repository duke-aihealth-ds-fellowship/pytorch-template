import numpy as np
import polars as pl
import torch
from captum.attr import (
    LayerGradientShap,
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer,
)
from torch.nn.utils.rnn import pad_sequence

from template.config import Config
from template.dataset import DataLoaders  # , collate_fn
from template.model import EmbeddingModel
from template.tune import load_best_checkpoint


def make_attributions(
    inputs: torch.Tensor, baselines: torch.Tensor, model: EmbeddingModel, cfg: Config
):
    inputs = inputs.to(cfg.trainer.device)
    baselines = baselines.to(cfg.trainer.device)
    model.to(cfg.trainer.device)
    interpretable_embedding = configure_interpretable_embedding_layer(
        model, "embeddings"
    )

    def forward(inputs):
        outputs = model(inputs)
        return outputs.sum(dim=-1)

    lig = LayerGradientShap(forward, layer=model.embeddings)  # type: ignore
    input_embeds = interpretable_embedding.indices_to_embeddings(inputs)
    baseline_embeds = interpretable_embedding.indices_to_embeddings(baselines)
    attributions, delta = lig.attribute(
        inputs=input_embeds, baselines=baseline_embeds, return_convergence_delta=True
    )
    remove_interpretable_embedding_layer(model, interpretable_embedding)
    print("Mean convergence delta:", delta.mean().item())
    # sum attributions across embedding dimension
    return attributions.sum(dim=-1)  # type: ignore


def format_attributions(attributions, inputs):
    df = pl.DataFrame(
        {
            "instance": np.repeat(np.arange(inputs.size(0)), inputs.size(1)),
            "position": np.tile(np.arange(inputs.shape[1]), inputs.shape[0]),
            "index": inputs.flatten().cpu().numpy(),
            "attribution": attributions.detach().flatten().cpu().numpy(),
        }
    )  # TODO .with_columns(pl.col("index").replace_strict(vocab_decoder).alias("token"))
    print(df)
    return df


def sum_attributions(df: pl.DataFrame):
    # sum attributions and count tokens within instances
    df = (
        df.group_by("instance", "index")
        .agg(pl.col("attribution").sum(), pl.col("token").count().alias("count"))
        .sort("instance", "index")
    )
    return df


# TODO decode tokens and sum attributions grouped by offsets
def feature_importance(cfg: Config, loaders: DataLoaders):
    model = load_best_checkpoint(cfg=cfg, model_class=EmbeddingModel)
    data = [instance["input_ids"] for instance in loaders.test.dataset]
    data = data[: cfg.importance.num_samples]
    data = pad_sequence(data, batch_first=True, padding_value=cfg.model.padding_idx)
    half = data.size(0) // 2
    inputs = data[:half]
    baselines = data[half:]
    attributions = make_attributions(
        inputs=inputs, baselines=baselines, model=model, cfg=cfg
    )
    df = format_attributions(attributions=attributions, inputs=inputs)
    df = sum_attributions(df)
    return df
