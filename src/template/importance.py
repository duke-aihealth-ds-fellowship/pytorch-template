import numpy as np
import polars as pl
import torch
from captum.attr import (
    LayerGradientShap,
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer,
)
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from template.config import Config
from template.model import EmbeddingModel
from template.tune import load_best_checkpoint


def make_attributions(
    target: int, inputs: torch.Tensor, baselines: torch.Tensor, cfg: Config
):
    model = load_best_checkpoint(cfg=cfg, model_class=EmbeddingModel)
    inputs = inputs.to(cfg.trainer.device)
    baselines = baselines.to(cfg.trainer.device)
    model.to(cfg.trainer.device)
    interpretable_embedding = configure_interpretable_embedding_layer(
        model, "embeddings"
    )
    lig = LayerGradientShap(model, layer=model.embeddings)  # type: ignore
    input_embeds = interpretable_embedding.indices_to_embeddings(inputs)
    baseline_embeds = interpretable_embedding.indices_to_embeddings(baselines)
    attributions, delta = lig.attribute(
        inputs=input_embeds,
        baselines=baseline_embeds,
        return_convergence_delta=True,
        target=target,
    )
    remove_interpretable_embedding_layer(model, interpretable_embedding)
    print("Mean convergence delta:", delta.mean().item())
    # sum attributions across embedding dimension
    attributions = attributions.sum(dim=-1)  # type: ignore
    return attributions


def format_attributions(
    text: list[str],
    word_ids: list[int],  # FIXME unused
    attributions: torch.Tensor,
    offsets: torch.Tensor,
):
    batch_size, seq_len = attributions.size()
    flat_offsets = offsets.flatten(0, 1).cpu().numpy()
    df = pl.DataFrame({"text": text}).with_row_index("sample_id")
    df = (
        pl.DataFrame(
            {
                "sample_id": np.repeat(np.arange(batch_size), seq_len),
                "start": flat_offsets[:, 0],
                "end": flat_offsets[:, 1],
                "attribution": attributions.flatten(0, 1).cpu().numpy(),
            }
        )
        .filter(pl.col("start") != pl.col("end"))
        .join(df, on="sample_id")
        .with_columns(
            word=pl.col("text").str.slice(
                pl.col("start"), pl.col("end") - pl.col("start")
            )
        )
        .group_by(["sample_id", "word"])
        .agg(pl.col("attribution").sum(), pl.col("word").count().alias("count"))
    )
    return df


def feature_importance(cfg: Config, loader: DataLoader):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.tokenizer.path)
    encodings = tokenizer(
        loader.dataset["text"],
        return_tensors="pt",
        padding=True,
        return_offsets_mapping=True,
    )
    n = cfg.attribution.num_samples
    offsets = encodings.offset_mapping[:n]
    input_ids = encodings.input_ids[:n]
    baselines = encodings.input_ids[n:]
    text = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    dfs: list[pl.DataFrame] = []
    for label in loader.dataset["label"].unique():
        attributions = make_attributions(
            target=label, inputs=input_ids, baselines=baselines, cfg=cfg
        )
        word_ids = [encodings.word_ids(i) for i in range(n)]
        df = format_attributions(
            text=text, word_ids=word_ids, attributions=attributions, offsets=offsets
        )
        df = df.with_columns(pl.lit(label).alias("label"))
        dfs.append(df)
    df = pl.concat(dfs)
    df.write_parquet(cfg.attribution.path)
    return df
