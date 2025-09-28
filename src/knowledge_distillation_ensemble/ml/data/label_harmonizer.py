"""
Label Harmonizer Module for Cybersecurity Datasets.

This module provides functionality for harmonizing labels across different
cybersecurity datasets, enabling consistent representation for cross-dataset
learning.
"""

import polars as pl
from pathlib import Path
from typing import Union
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("LabelHarmonizer")

# Define label mappings based on analysis
LABEL_MAPPINGS = {
    # CICIOT2023 -> Standard Labels
    "ciciot2023": {
        # Benign traffic
        "BenignTraffic": "Benign",
        # DDoS attacks (all variants)
        "DDoS-ACK_Fragmentation": "DDoS",
        "DDoS-HTTP_Flood": "DDoS",
        "DDoS-ICMP_Flood": "DDoS",
        "DDoS-ICMP_Fragmentation": "DDoS",
        "DDoS-PSHACK_Flood": "DDoS",
        "DDoS-RSTFINFlood": "DDoS",
        "DDoS-SYN_Flood": "DDoS",
        "DDoS-SlowLoris": "DDoS",
        "DDoS-SynonymousIP_Flood": "DDoS",
        "DDoS-TCP_Flood": "DDoS",
        "DDoS-UDP_Flood": "DDoS",
        "DDoS-UDP_Fragmentation": "DDoS",
        # DoS attacks
        "DoS-HTTP_Flood": "DoS",
        "DoS-SYN_Flood": "DoS",
        "DoS-TCP_Flood": "DoS",
        "DoS-UDP_Flood": "DoS",
        # Mirai botnet attacks (separate category)
        "Mirai-greeth_flood": "Mirai",
        "Mirai-greip_flood": "Mirai",
        "Mirai-udpplain": "Mirai",
        # Brute force related
        "DictionaryBruteForce": "Brute_Force",
        # Spoofing attacks
        "DNS_Spoofing": "Spoofing",
        "MITM-ArpSpoofing": "Spoofing",
        # Reconnaissance attacks (moved from Other)
        "VulnerabilityScan": "Reconnaissance",
        # Web-based attacks (keep as Other for now)
        "CommandInjection": "Other",
        "SqlInjection": "Other",
        "XSS": "Other",
        "Uploading_Attack": "Other",
        # Malware attacks (keep as Other for now)
        "Backdoor_Malware": "Other",
        "BrowserHijacking": "Other",
        # Reconnaissance attacks
        "Recon-HostDiscovery": "Reconnaissance",
        "Recon-OSScan": "Reconnaissance",
        "Recon-PingSweep": "Reconnaissance",
        "Recon-PortScan": "Reconnaissance",
    },
    # CICDIAD2024 -> Standard Labels
    "cicdiad2024": {
        "Benign": "Benign",
        "Brute_Force": "Brute_Force",
        "DDoS": "DDoS",
        "DoS": "DoS",
        "Mirai": "Mirai",  # Keep Mirai as separate category
        "Reconnaissance": "Reconnaissance",
        "Spoofing": "Spoofing",
    },
}

# Standard labels that will be used across all datasets
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


class LabelHarmonizer:
    """Class for harmonizing labels across cybersecurity datasets."""

    def __init__(self, default_label: str = "Unknown"):
        """
        Initialize the label harmonizer.

        Args:
            default_label: Label to use for unrecognized values
        """
        self.default_label = default_label
        self.label_mappings = LABEL_MAPPINGS
        self.standard_labels = STANDARD_LABELS
        self.logger = logger

    def harmonize_labels(
        self, df: pl.DataFrame, dataset_name: str, label_column: str = "label"
    ) -> pl.DataFrame:
        """
        Harmonize labels in a dataset according to predefined mappings.

        Args:
            df: The input dataframe with labels to harmonize
            dataset_name: Name of the dataset (e.g., 'ciciot2023',
                         'cicdiad2024')
            label_column: Column name containing the labels

        Returns:
            DataFrame with harmonized labels
        """
        if dataset_name not in self.label_mappings:
            self.logger.warning(f"No label mappings found for dataset: {dataset_name}")
            return df

        # Create a copy to avoid modifying the original dataframe
        df_harmonized = df.clone()

        # Get mapping for this dataset
        mapping = self.label_mappings[dataset_name]

        # Add harmonized label column using map_elements
        def map_label(label_val):
            """Map individual label values."""
            return mapping.get(label_val, self.default_label)

        df_harmonized = df_harmonized.with_columns(
            pl.col(label_column)
            .map_elements(map_label, return_dtype=pl.String)
            .alias("harmonized_label")
        )

        # Log label distribution before and after harmonization
        orig_counts = (
            df_harmonized.group_by(label_column).count().sort("count", descending=True)
        )

        harm_counts = (
            df_harmonized.group_by("harmonized_label")
            .count()
            .sort("count", descending=True)
        )

        self.logger.info(f"Original label distribution for {dataset_name}:")
        self.logger.info(f"{orig_counts}")

        self.logger.info(f"Harmonized label distribution for {dataset_name}:")
        self.logger.info(f"{harm_counts}")

        return df_harmonized

    def replace_original_labels(
        self, df: pl.DataFrame, remove_original: bool = False
    ) -> pl.DataFrame:
        """
        Replace the original label column with the harmonized labels.

        Args:
            df: DataFrame with 'harmonized_label' column
            remove_original: Whether to remove the original label column

        Returns:
            DataFrame with updated labels
        """
        if "harmonized_label" not in df.columns:
            self.logger.error("No 'harmonized_label' column found in dataframe")
            return df

        # Create a copy to avoid modifying the original dataframe
        result_df = df.clone()

        # Find the original label column (could be "Label" or "label")
        original_label_col = None
        for col in result_df.columns:
            if col.lower() in ["label", "class", "target"]:
                original_label_col = col
                break

        if not original_label_col:
            self.logger.error("Could not find original label column")
            return result_df

        # Create a new column 'original_label' with the original labels
        if not remove_original:
            result_df = result_df.with_columns(
                pl.col(original_label_col).alias("original_label")
            )

        # Replace the original label column with the harmonized labels
        result_df = result_df.with_columns(
            pl.col("harmonized_label").alias(original_label_col)
        )

        # Remove the harmonized_label column since it's now in the label column
        result_df = result_df.drop("harmonized_label")

        return result_df

    def harmonize_dataset_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        dataset_name: str,
        label_column: str = "label",
        remove_original: bool = False,
    ) -> None:
        """
        Harmonize labels in a parquet file and save to a new location.

        Args:
            input_path: Path to the input parquet file
            output_path: Path to save the harmonized parquet file
            dataset_name: Name of the dataset (e.g., 'ciciot2023',
                         'cicdiad2024')
            label_column: Column name containing the labels
            remove_original: Whether to remove the original label column
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Log the operation
        self.logger.info(f"Harmonizing labels in {input_path}")

        try:
            # Read the input file
            df = pl.read_parquet(input_path)

            # Harmonize labels
            df_harmonized = self.harmonize_labels(df, dataset_name, label_column)

            # Replace original labels if requested
            df_harmonized = self.replace_original_labels(df_harmonized, remove_original)

            # Save the harmonized dataset
            df_harmonized.write_parquet(output_path)

            self.logger.info(f"Harmonized dataset saved to {output_path}")

            # Calculate and log label distribution statistics
            total_rows = len(df_harmonized)

            # Find the label column name (should be the original name)
            label_col_for_stats = None
            for col in df_harmonized.columns:
                if col.lower() in ["label", "class", "target"]:
                    label_col_for_stats = col
                    break

            if label_col_for_stats:
                label_stats = (
                    df_harmonized.group_by(label_col_for_stats)
                    .agg(
                        pl.count().alias("count"),
                        (pl.count() / total_rows * 100).alias("percentage"),
                    )
                    .sort("count", descending=True)
                )
            else:
                label_stats = None

            self.logger.info("Harmonized label distribution statistics:")
            self.logger.info(f"{label_stats}")

            return label_stats

        except Exception as e:
            self.logger.error(f"Error harmonizing dataset: {e}")
            raise

    def get_label_distribution(
        self, df: pl.DataFrame, label_column: str = "label"
    ) -> pl.DataFrame:
        """
        Get the distribution of labels in a dataframe.

        Args:
            df: Input dataframe
            label_column: Column name containing the labels

        Returns:
            DataFrame with label counts and percentages
        """
        total_rows = len(df)

        return (
            df.group_by(label_column)
            .agg(
                pl.count().alias("count"),
                (pl.count() / total_rows * 100).alias("percentage"),
            )
            .sort("count", descending=True)
        )


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    from src.knowledge_distillation_ensemble.config.settings import settings

    # Initialize harmonizer
    harmonizer = LabelHarmonizer()

    # Define paths
    ciciot_path = settings.parquet_path / "ciciot2023_combined.parquet"
    cicdiad_path = settings.parquet_path / "cicdiad2024_combined.parquet"

    # Define output paths
    ciciot_harmonized = "ciciot2023_harmonized.parquet"
    cicdiad_harmonized = "cicdiad2024_harmonized.parquet"

    ciciot_harmonized_path = settings.parquet_path / ciciot_harmonized
    cicdiad_harmonized_path = settings.parquet_path / cicdiad_harmonized

    # Harmonize datasets
    ciciot_stats = harmonizer.harmonize_dataset_file(
        ciciot_path, ciciot_harmonized_path, "ciciot2023", remove_original=False
    )

    cicdiad_stats = harmonizer.harmonize_dataset_file(
        cicdiad_path, cicdiad_harmonized_path, "cicdiad2024", remove_original=False
    )

    print("Label harmonization complete!")
    print(f"CICIOT2023 harmonized labels saved to: {ciciot_harmonized_path}")
    print(f"CICDIAD2024 harmonized labels saved to: {cicdiad_harmonized_path}")
