import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib import colors

from template.config import Config, get_config


def plot_metrics(cfg: Config):
    df = pl.read_parquet(cfg.path.metrics)
    plt.figure()
    sns.set_theme(style=cfg.plots.style, font_scale=cfg.plots.font_scale)
    g = sns.catplot(
        data=df,
        x="metric",
        y="value",
        kind="bar",
        legend=True,
        errorbar="pi",  # 95% confidence interval
    )
    g.set_axis_labels("Metric", "Value")
    plt.savefig(cfg.path.metrics_plot)


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
    plt.figure()
    sns.set_theme(style=cfg.plots.style, font_scale=cfg.plots.font_scale)
    df = pl.read_parquet(cfg.path.attribution)
    df = get_top_k_tokens(df, k=10)
    vmin = min(df["count"].to_list())
    vmax = max(df["count"].to_list())
    norm = colors.Normalize(vmin, vmax)
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
    plt.savefig(cfg.path.importance_plot)


def plot(cfg: Config):
    plot_metrics(cfg)
    plot_attributions(cfg)


if __name__ == "__main__":
    cfg = get_config()
    plot(cfg)
