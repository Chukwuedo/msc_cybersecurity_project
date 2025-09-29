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
from matplotlib import ticker as mtick


# ---- Seaborn setup --------------------------------------------------------


try:
    # Optional: standard label order for consistency
    from src.knowledge_distillation_ensemble.ml.data.label_harmonizer import (
        STANDARD_LABELS,
    )
except Exception:  # pragma: no cover
    STANDARD_LABELS = [
        "Benign",
        "DDoS",
        "DoS",
        "Brute_Force",
        "Mirai",
        "Reconnaissance",
        "Spoofing",
        "Other",
        "Unknown",
    ]


def seaborn_init() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11


# ---- Label distributions --------------------------------------------------


def compute_label_stats(parquet_path: Path | str) -> pl.DataFrame:
    lf = pl.scan_parquet(str(parquet_path)).select("label")
    counts = lf.group_by("label").count().rename({"count": "n"}).collect()
    total = counts["n"].sum()
    return counts.with_columns((pl.col("n") / total * 100).alias("pct")).sort(
        "n", descending=True
    )


def _order_labels(labels: List[str]) -> List[str]:
    # Keep present labels; apply preferred order
    pref = {lab: i for i, lab in enumerate(STANDARD_LABELS)}
    return sorted(labels, key=lambda x: pref.get(x, 1_000 + hash(x) % 1_000))


def chart_label_stats(stats: pl.DataFrame, title: str):
    df = stats.to_pandas()
    order = _order_labels(df["label"].tolist())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)

    sns.barplot(ax=axes[0], data=df, x="label", y="n", order=order, color="#4C78A8")
    axes[0].set_title(f"{title} - counts")
    axes[0].set_xlabel("label")
    axes[0].set_ylabel("count")
    plt.setp(axes[0].get_xticklabels(), rotation=35, ha="right")
    axes[0].yaxis.set_major_formatter(
        mtick.FuncFormatter(
            lambda x, pos: f"{x / 1e6:.1f}M" if x >= 1e6 else f"{x / 1e3:.0f}K"
        )
    )

    sns.barplot(ax=axes[1], data=df, x="label", y="pct", order=order, color="#72B7B2")
    axes[1].set_title(f"{title} - percent")
    axes[1].set_xlabel("label")
    axes[1].set_ylabel("%")
    axes[1].set_ylim(0, 100)
    plt.setp(axes[1].get_xticklabels(), rotation=35, ha="right")

    # Annotate percent bars with values (top-3 only for clarity)
    top3 = df.sort_values("pct", ascending=False).head(3)
    for _, row in top3.iterrows():
        idx = order.index(row["label"]) if row["label"] in order else None
        if idx is not None:
            axes[1].text(
                idx,
                min(row["pct"] + 2.0, 98.0),
                f"{row['pct']:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#22504F",
            )

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
    # Order axes for readability
    df = df.reindex(index=_order_labels(list(df.index)))
    df = df[sorted(df.columns)]

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    sns.heatmap(
        df,
        ax=ax,
        cmap="viridis",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "%"},
        annot=df.where(df >= 1).notna(),
        fmt="",
    )
    # Add text annotations for cells >=1%
    for y, row in enumerate(df.index):
        for x, col in enumerate(df.columns):
            val = df.loc[row, col]
            if (isinstance(val, (int, float, np.floating))) and (val >= 1):
                ax.text(
                    x + 0.5,
                    y + 0.5,
                    f"{float(val):.1f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )

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
    return pl.DataFrame(
        out,
        schema=["dataset", "feature", "median", "q1", "q3"],
    )


def chart_feature_summary(summary: pl.DataFrame, title: str):
    df = summary.to_pandas()
    df["low"], df["high"] = df["q1"], df["q3"]

    # Robust x-limits per overall range (avoid long tails dominating)
    x_min = max(0.0, float(df["low"].min()))
    x_max = float(df["high"].quantile(0.95)) * 1.1

    fig, ax = plt.subplots(
        figsize=(10, max(3, 0.6 * df["feature"].nunique())),
        constrained_layout=True,
    )

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
    ax.set_xlim(x_min, x_max)
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

    # Ensure consistent dtype for concatenation and histogramming
    da = da.with_columns(pl.col(feature).cast(pl.Float64))
    db = db.with_columns(pl.col(feature).cast(pl.Float64))

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
        vals_arr = df.select(feature).to_series().to_numpy()
        vals = vals_arr.astype(float, copy=False)
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
    pdf = df.sort("ks", descending=True).to_pandas()

    # Dynamic width to fit many feature labels
    width = max(10, 0.45 * len(pdf["feature"]))
    fig, ax = plt.subplots(figsize=(width, 4), constrained_layout=True)
    sns.barplot(ax=ax, data=pdf, x="feature", y="ks", color="#F58518")
    ax.set_title(title)
    ax.set_xlabel("feature")
    ax.set_ylabel("KS distance")
    ax.set_ylim(0, 1)
    for i, v in enumerate(pdf["ks"].tolist()):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    # rotate labels to reduce clutter
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    # footnote about sampling
    fig.text(
        0.5,
        -0.02,
        "Sampled n=40k per dataset per feature",
        ha="center",
        va="top",
        fontsize=9,
        color="#555",
    )
    plt.show()
    return fig


# ---- Median delta (%) across datasets -------------------------------------


def compute_median_delta(
    path_a: Path | str,
    path_b: Path | str,
    features: Iterable[str],
    name_a: str = "CICIOT2023",
    name_b: str = "CICDIAD2024",
) -> pl.DataFrame:
    """Compute per-feature medians on both datasets and the delta (%).
    Δ% is computed as ((median_b - median_a) / max(|median_a|, 1e-12)) * 100.
    """
    fa = list(features)
    fb = list(features)
    da = _safe_sample(path_a, fa, 80_000)
    db = _safe_sample(path_b, fb, 80_000)

    rows = []
    for f in features:
        ma = float(da.select(pl.col(f).median()).item())
        mb = float(db.select(pl.col(f).median()).item())
        denom = max(abs(ma), 1e-12)
        delta_pct = ((mb - ma) / denom) * 100.0
        rows.append((f, ma, mb, delta_pct))

    return pl.DataFrame(
        rows,
        schema=[
            "feature",
            f"{name_a}_median",
            f"{name_b}_median",
            "delta_pct",
        ],
    ).sort("delta_pct", descending=True)


def chart_median_delta(df: pl.DataFrame, title: str):
    pdf = df.sort("delta_pct", descending=True).to_pandas()
    width = max(10, 0.45 * len(pdf["feature"]))
    fig, ax = plt.subplots(figsize=(width, 4), constrained_layout=True)
    sns.barplot(ax=ax, data=pdf, x="feature", y="delta_pct", color="#E45756")
    ax.set_title(title)
    ax.set_xlabel("feature")
    ax.set_ylabel("Δ median % (2024 vs 2023)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    # annotate
    for i, v in enumerate(pdf["delta_pct"].tolist()):
        offset = 0.02 * (1 if v >= 0 else -1)
        ax.text(
            i,
            v + offset,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.show()
    return fig


def median_delta_table(df: pl.DataFrame, top: int = 10) -> pl.DataFrame:
    """Return a compact table of top-|Δ%| features with both medians."""
    return (
        df.with_columns(pl.col("delta_pct").abs().alias("abs_delta"))
        .sort("abs_delta", descending=True)
        .drop("abs_delta")
        .head(top)
    )


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
    """Compute and render KS shift between two datasets (compact)."""
    if not path_a or not path_b:
        print("Need both harmonized datasets for KS comparison.")
        return None

    pa, pb = Path(path_a), Path(path_b)
    if not (pa.exists() and pb.exists()):
        print("Need both harmonized datasets for KS comparison.")
        return None

    ks_df = compute_ks_table(str(pa), str(pb), list(features))
    return chart_ks_table(ks_df, title)
