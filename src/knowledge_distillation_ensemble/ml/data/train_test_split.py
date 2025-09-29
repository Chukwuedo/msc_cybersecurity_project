"""
Stratified train/test splitting utilities using lazy evaluation.

- Returns Polars LazyFrames for train and test partitions
- Stratified by a label column (default: "label_multiclass")
- Deterministic and reproducible via a user-provided seed
- No new parquet files are created; splits are views over the analysis parquet
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import polars as pl

from .analysis_builder import create_analysis_parquet


DEFAULT_DATASET = "ciciot2023"
DEFAULT_LABEL_COL = "label_multiclass"


@dataclass(frozen=True)
class StratifiedSplit:
    train: pl.LazyFrame
    test: pl.LazyFrame


def _analysis_parquet_path(
    dataset_name: str = DEFAULT_DATASET,
    seed: int = 42,
) -> Path:
    """Ensure the analysis parquet exists and return its path.

    This will not overwrite existing files and is deterministic with the seed
    used by create_analysis_parquet for sampling, if generation is needed.
    """
    ds = dataset_name.lower()
    # Reuse/create analysis parquet if missing
    return create_analysis_parquet(ds, seed=seed, overwrite=False)


def stratified_split_lazy(
    parquet_path: Path | str,
    label_col: str = DEFAULT_LABEL_COL,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> StratifiedSplit:
    """Create a deterministic stratified train/test split as lazy views.

    The split is computed by hashing each row (seeded), ranking within each
    label group by the hash, and selecting the first `train_ratio` fraction for
    the train partition (per class), with the rest going to the test partition.

    No data is materialized until `.collect()` is called.
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1")

    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}")

    lf = pl.scan_parquet(str(path))

    # Seeded row hash for deterministic pseudo-random ordering per row
    # Using all columns maintains stability without relying on file order.
    lf = lf.with_columns(pl.struct(pl.all()).hash(seed=seed).alias("__row_hash__"))

    # Rank and count per class for stratification
    lf = lf.with_columns(
        pl.col("__row_hash__").rank(method="ordinal").over(label_col).alias("__rnk__"),
        pl.count().over(label_col).alias("__cnt__"),
    )

    # Threshold position for the train slice per class
    lf = lf.with_columns(
        (pl.col("__cnt__") * pl.lit(float(train_ratio))).alias("__thr__")
    )
    lf = lf.with_columns((pl.col("__rnk__") <= pl.col("__thr__")).alias("__is_train__"))

    # Build partitions (lazy views), dropping helper columns
    drop_cols = [
        "__row_hash__",
        "__rnk__",
        "__cnt__",
        "__thr__",
        "__is_train__",
    ]
    train_lf = lf.filter(pl.col("__is_train__")).drop(drop_cols)
    test_lf = lf.filter(~pl.col("__is_train__")).drop(drop_cols)

    return StratifiedSplit(train=train_lf, test=test_lf)


def get_stratified_split_lazy(
    dataset_name: str = DEFAULT_DATASET,
    label_col: str = DEFAULT_LABEL_COL,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> StratifiedSplit:
    """Convenience API to get stratified train/test lazy views for a dataset."""
    analysis_path = _analysis_parquet_path(dataset_name, seed=seed)
    return stratified_split_lazy(
        analysis_path,
        label_col=label_col,
        train_ratio=train_ratio,
        seed=seed,
    )


def compute_split_distributions(
    split: StratifiedSplit, label_col: str = DEFAULT_LABEL_COL
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Return counts and percent per class for train and test (materialized)."""

    def _stats(lf: pl.LazyFrame) -> pl.DataFrame:
        df = (
            lf.select(label_col)
            .group_by(label_col)
            .count()
            .rename({"count": "n"})
            .sort(label_col)
            .collect()
        )
        tot = int(df["n"].sum()) if df.height else 0
        if tot > 0:
            df = df.with_columns((pl.col("n") / pl.lit(tot) * 100).alias("pct"))
        else:
            df = df.with_columns(pl.lit(0.0).alias("pct"))
        return df

    return _stats(split.train), _stats(split.test)
