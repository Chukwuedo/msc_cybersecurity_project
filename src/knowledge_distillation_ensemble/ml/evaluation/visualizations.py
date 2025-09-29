"""
Aggregate visualizations for sourcing & preprocessing using seaborn/matplotlib.
Small, interpretable charts based on summary statistics and small samples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# ---- Seaborn setup --------------------------------------------------------


def seaborn_init() -> None:
    sns.set_theme(style="whitegrid", context="talk")


# ---- Label distributions --------------------------------------------------


def compute_label_stats(parquet_path: Path | str) -> pl.DataFrame:
    lf = pl.scan_parquet(str(parquet_path)).select("label")
    counts = lf.group_by("label").count().rename({"count": "n"}).collect()
    total = counts["n"].sum()
    return counts.with_columns((pl.col("n") / total * 100).alias("pct")).sort(
        "n", descending=True
    )


def chart_label_stats(stats: pl.DataFrame, title: str):
    df = stats.to_pandas()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    sns.barplot(ax=axes[0], data=df, x="label", y="n", color="#4C78A8")
    axes[0].set_title(f"{title} - counts")
    axes[0].set_xlabel("label")
    axes[0].set_ylabel("count")
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right")

    sns.barplot(ax=axes[1], data=df, x="label", y="pct", color="#72B7B2")
    axes[1].set_title(f"{title} - percent")
    axes[1].set_xlabel("label")
    axes[1].set_ylabel("%")
    axes[1].set_ylim(0, 100)
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")

    plt.show()
    return fig


# ---- Original -> harmonized mapping (CICIOT2023) -------------------------


def compute_mapping_percent(
    parquet_path: Path | str, top_original: int = 15
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    lf = pl.scan_parquet(str(parquet_path)).select(["original_label", "label"])
    ct = (
        lf.group_by(["original_label", "label"])
        .count()
        .rename({"count": "n"})
        .collect()
    )
    row_tot = ct.group_by("original_label").agg(pl.col("n").sum().alias("row_total"))
    ct_pct = (
        ct.join(row_tot, on="original_label", how="inner")
        .with_columns((pl.col("n") / pl.col("row_total") * 100).alias("pct"))
        .drop("row_total")
    )
    tops = (
        ct.group_by("original_label")
        .agg(pl.col("n").sum().alias("tot"))
        .sort("tot", descending=True)
        .head(top_original)
        .select("original_label")
    )
    ct_top = ct_pct.join(tops, on="original_label", how="inner")

    diag = ct_pct.filter(pl.col("original_label") == pl.col("label")).select(
        ["original_label", "pct"]
    )
    return ct_top, diag


def chart_mapping_heatmap(ct_top: pl.DataFrame, title: str):
    df = ct_top.to_pandas().pivot(index="label", columns="original_label", values="pct")
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    sns.heatmap(df, ax=ax, cmap="viridis", annot=False)
    ax.set_title(title)
    ax.set_xlabel("original")
    ax.set_ylabel("harmonized")
    plt.show()
    return fig


# ---- Feature summaries and shift -----------------------------------------


def _safe_sample(
    parquet_path: Path | str, columns: List[str], n: int = 40_000
) -> pl.DataFrame:
    lf = pl.scan_parquet(str(parquet_path)).select(columns)
    try:
        return lf.fetch(n)
    except Exception:
        return lf.limit(n).collect()


def compute_feature_summary(
    path: Path | str, features: Iterable[str], dataset_name: str
) -> pl.DataFrame:
    cols = list(features) + ["label"]
    df = _safe_sample(path, cols)
    out = []
    for f in features:
        s = df.select(
            pl.col(f).median().alias("median"),
            pl.col(f).quantile(0.25).alias("q1"),
            pl.col(f).quantile(0.75).alias("q3"),
        )
        m, q1, q3 = s.row(0)
        out.append((dataset_name, f, m, q1, q3))
    return pl.DataFrame(out, schema=["dataset", "feature", "median", "q1", "q3"])


def chart_feature_summary(summary: pl.DataFrame, title: str):
    df = summary.to_pandas()
    # Compute whiskers as IQR range
    df["low"] = df["q1"]
    df["high"] = df["q3"]

    fig, ax = plt.subplots(
        figsize=(10, max(3, 0.6 * df["feature"].nunique())), constrained_layout=True
    )
    # Horizontal ranges per dataset/feature
    for ds, sub in df.groupby("dataset"):
        ax.hlines(
            y=sub["feature"],
            xmin=sub["low"],
            xmax=sub["high"],
            label=f"{ds} IQR",
            linewidth=6,
        )
        ax.plot(sub["median"], sub["feature"], "o", label=f"{ds} median")
    ax.set_title(title)
    ax.set_xlabel("value")
    ax.set_ylabel("")
    ax.legend(loc="best", ncol=2)
    plt.show()
    return fig


def compute_ks(
    path_a: Path | str, path_b: Path | str, feature: str, n: int = 40_000
) -> float:
    da = _safe_sample(path_a, [feature], n).filter(pl.col(feature).is_finite())
    db = _safe_sample(path_b, [feature], n).filter(pl.col(feature).is_finite())
    if da.height == 0 or db.height == 0:
        return float("nan")

    pooled = pl.concat([da, db])
    s = pooled.select(
        pl.col(feature).min().alias("mn"),
        pl.col(feature).max().alias("mx"),
    )
    mn, mx = float(s[0, "mn"]), float(s[0, "mx"])
    if not (mx > mn):
        return 0.0

    bins = 60
    edges = np.linspace(mn, mx, num=bins + 1)

    def cdf_np(df: pl.DataFrame) -> np.ndarray:
        vals = df.select(feature).to_series().to_numpy()
        counts, _ = np.histogram(vals, bins=edges)
        total = counts.sum()
        if total == 0:
            return np.zeros(bins)
        return np.cumsum(counts) / total

    ca = cdf_np(da)
    cb = cdf_np(db)
    return float(np.max(np.abs(ca - cb)))


def compute_ks_table(
    path_a: Path | str, path_b: Path | str, features: Iterable[str]
) -> pl.DataFrame:
    rows = [(f, compute_ks(path_a, path_b, f)) for f in features]
    return pl.DataFrame(rows, schema=["feature", "ks"])


def chart_ks_table(df: pl.DataFrame, title: str):
    pdf = df.to_pandas()
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    sns.barplot(ax=ax, data=pdf, x="feature", y="ks", color="#F58518")
    ax.set_title(title)
    ax.set_xlabel("feature")
    ax.set_ylabel("KS distance")
    plt.show()
    return fig


# ---- Notebook-friendly wrappers ------------------------------------------


def show_feature_summary_for_datasets(
    path_a: Path | str | None,
    path_b: Path | str | None,
    features: Iterable[str],
    name_a: str = "CICIOT2023",
    name_b: str = "CICDIAD2024",
    title: str = "Feature medians (points) and IQR (bars)",
):
    """Compute and render feature medians/IQR for available datasets.

    Only small samples are used; renders a seaborn figure.
    """
    paths = []
    if path_a:
        p = Path(path_a)
        if p.exists():
            paths.append((p, name_a))
    if path_b:
        p = Path(path_b)
        if p.exists():
            paths.append((p, name_b))

    if not paths:
        print("Missing datasets for summary statistics.")
        return None

    summaries: List[pl.DataFrame] = []
    for p, nm in paths:
        summaries.append(compute_feature_summary(str(p), list(features), nm))

    if len(summaries) == 1:
        summary_df = summaries[0]
    else:
        summary_df = pl.concat(summaries, how="vertical_relaxed")
    return chart_feature_summary(summary_df, title)


def show_ks_between_datasets(
    path_a: Path | str | None,
    path_b: Path | str | None,
    features: Iterable[str],
    title: str = "Cross-dataset shift (KS distance)",
):
    """Compute and render KS shift for selected features between two datasets."""
    if not path_a or not path_b:
        print("Need both harmonized datasets for KS comparison.")
        return None

    pa, pb = Path(path_a), Path(path_b)
    if not (pa.exists() and pb.exists()):
        print("Need both harmonized datasets for KS comparison.")
        return None

    ks_df = compute_ks_table(str(pa), str(pb), list(features))
    return chart_ks_table(ks_df, title)
