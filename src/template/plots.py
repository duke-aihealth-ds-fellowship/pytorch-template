import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib import colors

from template.config import Config


def get_top_k_tokens(df: pl.DataFrame, k: int) -> pl.DataFrame:
    top_k = (
        df.group_by("label", "word")
        .agg(pl.col("attribution").mean().abs().alias("abs_attribution"))
        .select(
            pl.all()
            .top_k_by(by="abs_attribution", k=k)
            .over("label", mapping_strategy="explode")
        )
    )
    df = df.join(top_k, on=["label", "word"])
    df = df.sort("label", "abs_attribution", descending=True)
    return df


def plot_attributions(cfg: Config):
    sns.set_theme(style=cfg.plots.style, font_scale=cfg.plots.font_scale)
    plt.figure()
    df = pl.read_parquet(cfg.attribution.path)
    df = get_top_k_tokens(df, k=10)
    vmin = min(df["count"].to_list())
    vmax = max(df["count"].to_list())
    norm = colors.Normalize(vmin, vmax)
    df = df.with_columns(
        pl.col("label").replace_strict(
            {"0": "World", "1": "Sports", "2": "Business", "3": "Sci/Tech"}
        )
    )
    g = sns.catplot(
        data=df,
        x="attribution",
        y="word",
        hue=df["count"].to_list(),
        hue_norm=norm,
        col="label",
        col_wrap=2,
        kind="strip",
        linestyles="",
        palette=cfg.plots.palette,
        legend=False,
        jitter=0.2,
        alpha=0.5,
        sharey=False,
        sharex=False,
    )
    g.set_axis_labels("Shap value", "Word")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array(np.array([norm.vmin, norm.vmax]))
    color_bar = g.figure.colorbar(sm, ax=g.axes.ravel().tolist())
    color_bar.set_label("Count")
    for ax in g.axes.flat:
        ax.grid(axis="y")
        ax.axvline(0, color="black", linestyle="--")
    plt.savefig(cfg.plots.importance)
