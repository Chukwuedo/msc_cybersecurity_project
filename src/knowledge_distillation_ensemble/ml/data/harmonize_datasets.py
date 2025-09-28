"""Dataset harmonization module for IoT cybersecurity datasets.

This module provides functionality to harmonize CICIOT2023 and CIC DIAD 2024
datasets into a common schema for cross-dataset machine learning.
"""

from pathlib import Path
from typing import Optional
import polars as pl
import re

from ...config.settings import settings


class DatasetHarmonizer:
    """Harmonizes IoT cybersecurity datasets to a common schema."""

    # Canonical output schema (24 common features as per mapping analysis)
    COMMON_FEATURES = [
        "flow_duration",
        "flow_packets_per_second",
        "forward_packets_per_second",
        "backward_packets_per_second",
        "flow_iat_mean",
        "packet_length_min",
        "packet_length_max",
        "packet_length_mean",
        "packet_length_std",
        "packet_length_range",  # engineered: max - min
        "fin_flag_count",  # binarised 0/1
        "syn_flag_count",
        "rst_flag_count",
        "psh_flag_count",
        "ack_flag_count",
        "ece_flag_count",
        "cwr_flag_count",
        "urg_flag_count",
        "total_packets",  # composite (fwd + bwd)
        "total_bytes",  # composite (fwd + bwd)
        "average_packet_size",
        "flow_bytes_per_second",  # engineered: total_bytes / duration
        "header_length_total",  # composite (fwd + bwd)
        "label",
    ]

    def __init__(self, output_dir: Path):
        """Initialize harmonizer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_column_name(name: str) -> str:
        """Normalize column names for consistent matching."""
        s = name.strip()
        s = s.replace("/", "_per_")
        s = re.sub(r"[^\w]+", "_", s)  # spaces, hyphens → _
        s = re.sub(r"_+", "_", s)
        s = s.strip("_").lower()
        return s

    def _rename_columns(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Rename columns using normalized names."""
        return df.rename({c: self._normalize_column_name(c) for c in df.columns})

    def _first_present_column(
        self, df: pl.LazyFrame, *candidates: str
    ) -> Optional[str]:
        """Find first existing column from candidates."""
        cols = set(df.columns)
        for c in candidates:
            n = self._normalize_column_name(c)
            if n in cols:
                return n
        return None

    def _binarize_flags(self, df: pl.LazyFrame, *candidates: str) -> pl.Expr:
        """Convert flag counts to 0/1 binary values."""
        present = [
            self._normalize_column_name(c)
            for c in candidates
            if self._normalize_column_name(c) in df.columns
        ]
        if not present:
            return pl.lit(None)

        cols = [pl.col(c) for c in present]
        any_not_null = (
            pl.max_horizontal(*[c.is_not_null().cast(pl.Int8) for c in cols]) == 1
        )
        sum_pos = pl.sum_horizontal(*[pl.coalesce([c, pl.lit(0)]) for c in cols])
        return (
            pl.when(any_not_null)
            .then((sum_pos > 0).cast(pl.Int8))
            .otherwise(pl.lit(None))
        )

    def _safe_diff(self, a: Optional[pl.Expr], b: Optional[pl.Expr]) -> pl.Expr:
        """Safe subtraction handling None values."""
        if a is None or b is None:
            return pl.lit(None)
        return a - b

    def _safe_sum(self, *exprs: Optional[pl.Expr]) -> pl.Expr:
        """Safe sum handling None values."""
        valid_exprs = [e for e in exprs if e is not None]
        if not valid_exprs:
            return pl.lit(None)

        filled = [pl.coalesce([e, pl.lit(0)]) for e in valid_exprs]
        s = filled[0]
        for e in filled[1:]:
            s = s + e

        # Return null if all inputs were null
        any_not_null = (
            pl.max_horizontal(*[e.is_not_null().cast(pl.Int8) for e in valid_exprs])
            == 1
        )
        return pl.when(any_not_null).then(s).otherwise(pl.lit(None))

    def _safe_divide(
        self, numer: Optional[pl.Expr], denom: Optional[pl.Expr]
    ) -> pl.Expr:
        """Safe division handling None and zero values."""
        if numer is None or denom is None:
            return pl.lit(None)
        return (
            pl.when((denom.is_not_null()) & (denom > 0))
            .then(numer / denom)
            .otherwise(pl.lit(None))
        )

    def harmonize_ciciot2023(self, parquet_file: Optional[Path] = None) -> pl.LazyFrame:
        """Transform CICIOT2023 data to common schema."""
        if parquet_file is None:
            parquet_file = settings.parquet_path / "ciciot2023_combined.parquet"

        lf = pl.scan_parquet(parquet_file).pipe(self._rename_columns)

        # Map column names to normalized versions
        flow_duration = self._first_present_column(lf, "flow_duration")
        rate = self._first_present_column(lf, "rate")
        srate = self._first_present_column(lf, "srate")
        drate = self._first_present_column(lf, "drate")
        iat = self._first_present_column(lf, "iat")
        pmin = self._first_present_column(lf, "min")
        pmax = self._first_present_column(lf, "max")
        pmean = self._first_present_column(lf, "avg")
        pstd = self._first_present_column(lf, "std")
        tot_sum = self._first_present_column(lf, "tot sum")
        tot_size = self._first_present_column(lf, "tot size")
        header_len = self._first_present_column(lf, "header_length", "header length")
        number = self._first_present_column(lf, "number")
        label = self._first_present_column(lf, "label") or "label"

        # Helper for safe column reference
        c = lambda n: pl.col(n) if n else pl.lit(None)

        return lf.select(
            [
                c(flow_duration).alias("flow_duration"),
                c(rate).alias("flow_packets_per_second"),
                c(srate).alias("forward_packets_per_second"),
                c(drate).alias("backward_packets_per_second"),
                c(iat).alias("flow_iat_mean"),
                c(pmin).alias("packet_length_min"),
                c(pmax).alias("packet_length_max"),
                c(pmean).alias("packet_length_mean"),
                c(pstd).alias("packet_length_std"),
                self._safe_diff(c(pmax), c(pmin)).alias("packet_length_range"),
                # TCP Flags (binarized)
                self._binarize_flags(lf, "fin_flag_number", "fin_count").alias(
                    "fin_flag_count"
                ),
                self._binarize_flags(lf, "syn_flag_number", "syn_count").alias(
                    "syn_flag_count"
                ),
                self._binarize_flags(lf, "rst_flag_number", "rst_count").alias(
                    "rst_flag_count"
                ),
                self._binarize_flags(lf, "psh_flag_number").alias("psh_flag_count"),
                self._binarize_flags(lf, "ack_flag_number", "ack_count").alias(
                    "ack_flag_count"
                ),
                self._binarize_flags(lf, "ece_flag_number").alias("ece_flag_count"),
                self._binarize_flags(lf, "cwr_flag_number").alias("cwr_flag_count"),
                self._binarize_flags(lf, "urg_count").alias("urg_flag_count"),
                # Volume metrics
                c(number).alias("total_packets"),
                c(tot_sum).alias("total_bytes"),
                c(tot_size).alias("average_packet_size"),
                self._safe_divide(c(tot_sum), c(flow_duration)).alias(
                    "flow_bytes_per_second"
                ),
                c(header_len).alias("header_length_total"),
                c(label).cast(pl.Utf8).alias("label"),
            ]
        ).select(
            [
                pl.col(col) if col in self.COMMON_FEATURES else pl.lit(None).alias(col)
                for col in self.COMMON_FEATURES
            ]
        )

    def harmonize_cicdiad2024(
        self, parquet_file: Optional[Path] = None
    ) -> pl.LazyFrame:
        """Transform CIC DIAD 2024 data to common schema."""
        if parquet_file is None:
            parquet_file = settings.parquet_path / "cicdiad2024_combined.parquet"

        lf = pl.scan_parquet(parquet_file).pipe(self._rename_columns)

        # Map columns
        flow_duration = self._first_present_column(lf, "flow duration")
        flow_pkts_s = self._first_present_column(lf, "flow packets_s")
        fwd_pkts_s = self._first_present_column(lf, "fwd packets_s")
        bwd_pkts_s = self._first_present_column(lf, "bwd packets_s")
        flow_iat_mean = self._first_present_column(lf, "flow iat mean")

        pmin = self._first_present_column(lf, "packet length min")
        pmax = self._first_present_column(lf, "packet length max")
        pmean = self._first_present_column(lf, "packet length mean")
        pstd = self._first_present_column(lf, "packet length std")

        # TCP flags
        fin = self._first_present_column(lf, "fin flag count")
        syn = self._first_present_column(lf, "syn flag count")
        rst = self._first_present_column(lf, "rst flag count")
        psh = self._first_present_column(lf, "psh flag count")
        ack = self._first_present_column(lf, "ack flag count")
        ece = self._first_present_column(lf, "ece flag count")
        cwr = self._first_present_column(lf, "cwr flag count")
        urg = self._first_present_column(lf, "urg flag count")

        # Volume metrics
        total_fwd_pkts = self._first_present_column(lf, "total fwd packet")
        total_bwd_pkts = self._first_present_column(lf, "total bwd packets")
        total_len_fwd = self._first_present_column(lf, "total length of fwd packet")
        total_len_bwd = self._first_present_column(lf, "total length of bwd packet")
        avg_pkt_size = self._first_present_column(lf, "average packet size")
        flow_bytes_s = self._first_present_column(lf, "flow bytes_s")
        fwd_hdr_len = self._first_present_column(lf, "fwd header length")
        bwd_hdr_len = self._first_present_column(lf, "bwd header length")
        label = self._first_present_column(lf, "label") or "label"

        c = lambda n: pl.col(n) if n else pl.lit(None)

        return lf.select(
            [
                c(flow_duration).alias("flow_duration"),
                c(flow_pkts_s).alias("flow_packets_per_second"),
                c(fwd_pkts_s).alias("forward_packets_per_second"),
                c(bwd_pkts_s).alias("backward_packets_per_second"),
                c(flow_iat_mean).alias("flow_iat_mean"),
                c(pmin).alias("packet_length_min"),
                c(pmax).alias("packet_length_max"),
                c(pmean).alias("packet_length_mean"),
                c(pstd).alias("packet_length_std"),
                self._safe_diff(c(pmax), c(pmin)).alias("packet_length_range"),
                # TCP Flags (binarized)
                self._binarize_flags(lf, fin).alias("fin_flag_count"),
                self._binarize_flags(lf, syn).alias("syn_flag_count"),
                self._binarize_flags(lf, rst).alias("rst_flag_count"),
                self._binarize_flags(lf, psh).alias("psh_flag_count"),
                self._binarize_flags(lf, ack).alias("ack_flag_count"),
                self._binarize_flags(lf, ece).alias("ece_flag_count"),
                self._binarize_flags(lf, cwr).alias("cwr_flag_count"),
                self._binarize_flags(lf, urg).alias("urg_flag_count"),
                # Volume metrics (composite)
                self._safe_sum(c(total_fwd_pkts), c(total_bwd_pkts)).alias(
                    "total_packets"
                ),
                self._safe_sum(c(total_len_fwd), c(total_len_bwd)).alias("total_bytes"),
                c(avg_pkt_size).alias("average_packet_size"),
                pl.when(c(flow_bytes_s).is_not_null())
                .then(c(flow_bytes_s))
                .otherwise(
                    self._safe_divide(
                        self._safe_sum(c(total_len_fwd), c(total_len_bwd)),
                        c(flow_duration),
                    )
                )
                .alias("flow_bytes_per_second"),
                self._safe_sum(c(fwd_hdr_len), c(bwd_hdr_len)).alias(
                    "header_length_total"
                ),
                c(label).cast(pl.Utf8).alias("label"),
            ]
        ).select(
            [
                pl.col(col) if col in self.COMMON_FEATURES else pl.lit(None).alias(col)
                for col in self.COMMON_FEATURES
            ]
        )

    def process_datasets(
        self,
        ciciot2023_file: Optional[Path] = None,
        cicdiad2024_file: Optional[Path] = None,
    ) -> tuple[Path, Path]:
        """Process both datasets and save harmonized versions.

        Args:
            ciciot2023_file: Path to CICIOT2023 parquet file (optional)
            cicdiad2024_file: Path to CIC DIAD 2024 parquet file (optional)

        Returns:
            Tuple of paths to harmonized parquet files
        """
        print("🔄 Harmonizing CICIOT2023 dataset...")
        cic23_lf = self.harmonize_ciciot2023(ciciot2023_file)
        cic23_output = self.output_dir / "revised_ciciot2023.parquet"
        cic23_lf.sink_parquet(cic23_output, compression="zstd")

        print("🔄 Harmonizing CIC DIAD 2024 dataset...")
        diad_lf = self.harmonize_cicdiad2024(cicdiad2024_file)
        diad_output = self.output_dir / "revised_cicdiad2024.parquet"
        diad_lf.sink_parquet(diad_output, compression="zstd")

        print("✅ Harmonized datasets saved:")
        print(f"   📁 {cic23_output}")
        print(f"   📁 {diad_output}")
        print(f"   🎯 {len(self.COMMON_FEATURES)} common features")

        return cic23_output, diad_output

    def get_harmonized_preview(
        self, file_path: Path, n_rows: int = 5
    ) -> Optional[pl.DataFrame]:
        """Get preview of harmonized dataset."""
        try:
            if file_path and file_path.exists():
                df = pl.read_parquet(file_path)
                return df.head(n_rows)
            return None
        except Exception as e:
            print(f"Error reading harmonized dataset: {e}")
            return None
