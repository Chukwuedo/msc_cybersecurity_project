"""
Feature harmonizer for producing the 23 snake_case features
from raw CICIOT2023 combined parquet, plus label harmonization
and original_label preservation.
"""

from __future__ import annotations

from pathlib import Path
import polars as pl

from .label_harmonizer import LabelHarmonizer


class FeatureHarmonizer:
    """Builds the harmonized 23-feature schema + labels for supported
    datasets.
    """

    def __init__(self) -> None:
        self.labeler = LabelHarmonizer()

    def _exprs_ciciot2023(
        self,
        lf: pl.LazyFrame,
        label_column: str,
    ) -> list[pl.Expr]:
        def _coalesce(*cols: str) -> pl.Expr:
            return pl.coalesce(
                [pl.col(c) for c in cols if c in lf.columns] + [pl.lit(0)]
            )

        return [
            pl.col("Duration").alias("flow_duration"),
            pl.col("Rate").alias("flow_packets_per_second"),
            pl.col("Srate").alias("forward_packets_per_second"),
            pl.col("Drate").alias("backward_packets_per_second"),
            pl.col("IAT").alias("flow_iat_mean"),
            pl.col("Min").alias("packet_length_min"),
            pl.col("Max").alias("packet_length_max"),
            pl.col("AVG").alias("packet_length_mean"),
            pl.col("Std").alias("packet_length_std"),
            (pl.col("Max") - pl.col("Min")).alias("packet_length_range"),
            _coalesce("fin_flag_number", "fin_count").alias("fin_flag_count"),
            _coalesce("syn_flag_number", "syn_count").alias("syn_flag_count"),
            _coalesce("rst_flag_number", "rst_count").alias("rst_flag_count"),
            pl.col("psh_flag_number").alias("psh_flag_count"),
            _coalesce("ack_flag_number", "ack_count").alias("ack_flag_count"),
            pl.col("ece_flag_number").alias("ece_flag_count"),
            pl.col("cwr_flag_number").alias("cwr_flag_count"),
            pl.col("urg_count").alias("urg_flag_count"),
            pl.col("Number").alias("total_packets"),
            pl.col("Tot sum").alias("total_bytes"),
            pl.col("Tot size").alias("average_packet_size"),
            (pl.col("Tot sum") / (pl.col("Duration") + pl.lit(1e-9))).alias(
                "flow_bytes_per_second"
            ),
            pl.col("Header_Length").alias("header_length_total"),
            pl.col(label_column).alias("_raw_label"),
        ]

    def _exprs_cicdiad2024(
        self,
        _lf: pl.LazyFrame,
        label_column: str,
    ) -> list[pl.Expr]:
        return [
            pl.col("Flow Duration").alias("flow_duration"),
            pl.col("Flow Packets/s").alias("flow_packets_per_second"),
            pl.col("Fwd Packets/s").alias("forward_packets_per_second"),
            pl.col("Bwd Packets/s").alias("backward_packets_per_second"),
            pl.col("Flow IAT Mean").alias("flow_iat_mean"),
            pl.col("Packet Length Min").alias("packet_length_min"),
            pl.col("Packet Length Max").alias("packet_length_max"),
            pl.col("Packet Length Mean").alias("packet_length_mean"),
            pl.col("Packet Length Std").alias("packet_length_std"),
            (pl.col("Packet Length Max") - pl.col("Packet Length Min")).alias(
                "packet_length_range"
            ),
            pl.col("FIN Flag Count").alias("fin_flag_count"),
            pl.col("SYN Flag Count").alias("syn_flag_count"),
            pl.col("RST Flag Count").alias("rst_flag_count"),
            pl.col("PSH Flag Count").alias("psh_flag_count"),
            pl.col("ACK Flag Count").alias("ack_flag_count"),
            pl.col("ECE Flag Count").alias("ece_flag_count"),
            pl.col("CWR Flag Count").alias("cwr_flag_count"),
            pl.col("URG Flag Count").alias("urg_flag_count"),
            # Use component sums available in combined file
            (pl.col("Total Fwd Packet") + pl.col("Total Bwd packets")).alias(
                "total_packets"
            ),
            (
                pl.col("Total Length of Fwd Packet")
                + pl.col("Total Length of Bwd Packet")
            ).alias("total_bytes"),
            pl.col("Average Packet Size").alias("average_packet_size"),
            pl.col("Flow Bytes/s").alias("flow_bytes_per_second"),
            (pl.col("Fwd Header Length") + pl.col("Bwd Header Length")).alias(
                "header_length_total"
            ),
            pl.col(label_column).alias("_raw_label"),
        ]

    def harmonize_dataset(
        self,
        input_path: Path | str,
        output_path: Path | str,
        dataset_name: str,
        label_column: str = "label",
    ) -> None:
        """Create harmonized features and labels for the given dataset."""
        in_path = Path(input_path)
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        lf = pl.scan_parquet(str(in_path))

        if dataset_name == "ciciot2023":
            exprs = self._exprs_ciciot2023(lf, label_column)
        elif dataset_name == "cicdiad2024":
            # DIAD raw label column is typically "Label"
            label_col = label_column if label_column in lf.columns else "Label"
            exprs = self._exprs_cicdiad2024(lf, label_col)
        else:
            raise ValueError(f"Unsupported dataset_name: {dataset_name}")

        # Project features + raw label lazily
        base = lf.select(exprs)

        # Apply label harmonization lazily using central mappings
        mapping = self.labeler.label_mappings.get(dataset_name, {})
        default_label = self.labeler.default_label

        def _map_label(v: str) -> str:
            return mapping.get(v, default_label)

        final = base.with_columns(
            pl.col("_raw_label").alias("original_label"),
            pl.col("_raw_label")
            .map_elements(_map_label, return_dtype=pl.String)
            .alias("label"),
        ).drop("_raw_label")
        # Add requested label encodings
        final = final.with_columns(
            # Binary: 0 for Benign, 1 otherwise
            (pl.when(pl.col("label") == pl.lit("Benign")).then(0).otherwise(1))
            .cast(pl.Int8)
            .alias("label_binary"),
            # Multiclass mapping per specification; unknowns map to 7 (Other)
            (
                pl.when(pl.col("label") == pl.lit("Benign"))
                .then(0)
                .when(pl.col("label") == pl.lit("DDoS"))
                .then(1)
                .when(pl.col("label") == pl.lit("DoS"))
                .then(2)
                .when(pl.col("label") == pl.lit("Brute_Force"))
                .then(3)
                .when(pl.col("label") == pl.lit("Mirai"))
                .then(4)
                .when(pl.col("label") == pl.lit("Reconnaissance"))
                .then(5)
                .when(pl.col("label") == pl.lit("Spoofing"))
                .then(6)
                .otherwise(7)
            )
            .cast(pl.Int8)
            .alias("label_multiclass"),
        )

        # Write out lazily
        final.sink_parquet(str(out_path))

    # Back-compat API for existing calls
    def harmonize_ciciot2023(
        self,
        input_path: Path | str,
        output_path: Path | str,
        label_column: str = "label",
    ) -> None:
        self.harmonize_dataset(
            input_path,
            output_path,
            "ciciot2023",
            label_column,
        )
