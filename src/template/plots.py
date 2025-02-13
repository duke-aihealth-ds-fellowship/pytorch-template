import matplotlib.colors as colors
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from template.config import Config


def get_top_k_tokens(df: pl.DataFrame, top_k: int) -> pl.Series:
    return (
        df.group_by("token")
        .agg(pl.col("attribution").mean().abs())
        .sort("attribution")
        .tail(top_k)
    )["token"].reverse()


def plot_attributions(df: pl.DataFrame, cfg: Config):
    sns.set_theme(style="darkgrid", font_scale=cfg.plots.font_scale)
    top_k_tokens = get_top_k_tokens(df, top_k=20)
    ax = sns.stripplot(
        data=df,
        x="attribution",
        y="token",
        hue="count",
        linestyles="",
        order=top_k_tokens,
        palette=cfg.plots.palette,
        legend=False,
        jitter=0.2,
        alpha=0.5,
    )
    ax.set_ylabel("token")
    plt.grid(axis="y")
    plt.axvline(0, color="black", linestyle="--")
    norm = colors.Normalize(min(df["count"].to_list()), max(df["count"].to_list()))
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Count")
    plt.savefig(cfg.importance.plot_path, format="pdf")
