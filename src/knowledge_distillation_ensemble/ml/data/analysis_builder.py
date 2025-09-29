"""
Analysis dataset builder (renamed from analysis_dataset).

Creates balanced-ish analysis parquet datasets from the harmonized
(revised_*) parquets with the following characteristics per dataset:
- label_multiclass == 0 (Benign): up to 200,000 rows
- label_multiclass in {1..7}: up to 100,000 rows each
If available rows are fewer than the cap for a class, take all.

If the revised_* parquet is missing, it is generated from the combined
parquet using FeatureHarmonizer first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import duckdb

from .feature_harmonizer import FeatureHarmonizer
from ...config.settings import settings


_DATASETS: Dict[str, Dict[str, str]] = {
    "ciciot2023": {
        "combined": "ciciot2023_combined.parquet",
        "revised": "revised_ciciot2023.parquet",
        "analysis": "analysis_ciciot2023.parquet",
        "label_column": "label",
    },
    "cicdiad2024": {
        "combined": "cicdiad2024_combined.parquet",
        "revised": "revised_cicdiad2024.parquet",
        "analysis": "analysis_cicdiad2024.parquet",
        "label_column": "Label",
    },
}


def _ensure_revised(dataset_name: str) -> Path:
    ds = dataset_name.lower()
    if ds not in _DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    parquet_dir = settings.parquet_path
    processed_dir = settings.processed_parquet_path
    processed_dir.mkdir(parents=True, exist_ok=True)

    revised_path = processed_dir / _DATASETS[ds]["revised"]
    if revised_path.exists():
        return revised_path

    combined_path = parquet_dir / _DATASETS[ds]["combined"]
    if not combined_path.exists():
        raise FileNotFoundError(f"Missing combined parquet for {ds}: {combined_path}")

    label_col = _DATASETS[ds]["label_column"]
    fh = FeatureHarmonizer()
    fh.harmonize_dataset(str(combined_path), str(revised_path), ds, label_col)
    return revised_path


def create_analysis_parquet(
    dataset_name: str,
    seed: int = 42,
    overwrite: bool = False,
) -> Path:
    """Create an analysis parquet for the given dataset.

    Args:
        dataset_name: "ciciot2023" or "cicdiad2024"
        seed: Random seed for reproducible sampling
        overwrite: When False and output exists, skip re-creation

    Returns:
        Path to the created (or existing) analysis parquet
    """
    ds = dataset_name.lower()
    if ds not in _DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    revised_path = _ensure_revised(ds)
    out_path = settings.processed_parquet_path / _DATASETS[ds]["analysis"]
    if out_path.exists() and not overwrite:
        return out_path

    # Sampling caps
    caps = {0: 200_000}
    for k in range(1, 8):
        caps[k] = 100_000

    # Build per-class reservoir samples with REPEATABLE(seed)
    # and union them; finally shuffle deterministically and write.
    base = str(revised_path).replace("'", "''")

    class_queries = []
    for cls, cap in caps.items():
        # Use REPEATABLE with class-offset to vary sequence per class
        rep_seed = seed + cls * 9973
        q = (
            "SELECT * FROM ("
            "SELECT * FROM read_parquet('" + base + "') "
            f"WHERE label_multiclass = {cls}"
            ") t USING SAMPLE RESERVOIR ("
            + str(cap)
            + " ROWS) REPEATABLE ("
            + str(rep_seed)
            + ")"
        )
        class_queries.append(q)

    union_query = " UNION ALL ".join(class_queries)

    # Wrap union and write directly; REPEATABLE seeds ensure determinism
    with duckdb.connect() as con:
        final_query = union_query
        out_escaped = str(out_path).replace("'", "''")
        copy_stmt = (
            "COPY ("
            + final_query
            + ") TO '"
            + out_escaped
            + "' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        con.execute(copy_stmt)

    return out_path


def create_all_analysis_parquets(
    seed: int = 42, overwrite: bool = False
) -> Dict[str, Path]:
    """Create analysis parquets for all supported datasets."""
    results: Dict[str, Path] = {}
    for ds in _DATASETS.keys():
        results[ds] = create_analysis_parquet(
            ds,
            seed=seed,
            overwrite=overwrite,
        )
    return results
