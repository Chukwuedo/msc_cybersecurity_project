"""
Data extraction module for Knowledge Distillation Ensemble project.

This module handles downloading and preparing datasets for the project:
- CICIOT 2023: Main training and testing data
- CIC IOT DI-AD 2024: Unseen test data for robustness validation
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import kagglehub

from src.knowledge_distillation_ensemble.config.settings import settings

# Set up logging
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# Dataset identifiers
CICIOT2023_DATASET = "madhavmalhotra/unb-cic-iot-dataset"
CICDIAD2024_DATASET = "aymeneinformatique/cic-diad-new-2024"


class DatasetExtractor:
    """Handles dataset extraction and management for the project."""

    def __init__(self):
        self.data_dir = settings.data_path
        self.ciciot2023_path: Optional[Path] = None
        self.cicdiad2024_path: Optional[Path] = None

    def download_datasets(self) -> Dict[str, Path]:
        """
        Download both required datasets.

        Returns:
            Dict mapping dataset names to their download paths
        """
        logger.info("Checking dataset availability...")

        try:
            # Check and download CICIOT 2023 dataset
            ciciot2023_raw_path = kagglehub.dataset_download(CICIOT2023_DATASET)
            self.ciciot2023_path = Path(ciciot2023_raw_path)

            # Check if dataset was already cached
            if self._is_dataset_cached(self.ciciot2023_path):
                logger.info(
                    f"CICIOT 2023 dataset already exists at: {self.ciciot2023_path}"
                )
            else:
                logger.info(
                    f"Downloaded CICIOT 2023 dataset to: {self.ciciot2023_path}"
                )

            # Check and download CIC DIAD 2024 dataset
            cicdiad2024_raw_path = kagglehub.dataset_download(CICDIAD2024_DATASET)
            self.cicdiad2024_path = Path(cicdiad2024_raw_path)

            # Check if dataset was already cached
            if self._is_dataset_cached(self.cicdiad2024_path):
                logger.info(
                    f"CIC DIAD 2024 dataset already exists at: {self.cicdiad2024_path}"
                )
            else:
                logger.info(
                    f"Downloaded CIC DIAD 2024 dataset to: {self.cicdiad2024_path}"
                )

            logger.info("Dataset preparation completed successfully!")

            return {
                "ciciot2023": self.ciciot2023_path,
                "cicdiad2024": self.cicdiad2024_path,
            }

        except Exception as e:
            logger.error(f"Error preparing datasets: {e}")
            raise

    def _is_dataset_cached(self, dataset_path: Path) -> bool:
        """
        Check if dataset is already cached by looking for CSV files.

        Args:
            dataset_path: Path to the dataset directory

        Returns:
            True if dataset has CSV files, False otherwise
        """
        if not dataset_path.exists():
            return False

        # Check if there are any CSV files in the dataset
        csv_files = list(dataset_path.rglob("*.csv"))
        return len(csv_files) > 0

    def get_ciciot2023_csv_files(self) -> list[Path]:
        """
        Get all CSV files from the CICIOT 2023 dataset.

        Returns:
            List of CSV file paths
        """
        if not self.ciciot2023_path:
            raise ValueError(
                "CICIOT 2023 dataset not downloaded yet. "
                "Call download_datasets() first."
            )

        csv_files = list(self.ciciot2023_path.rglob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files in CICIOT 2023 dataset")
        return csv_files

    def get_cicdiad2024_csv_files(self) -> list[Path]:
        """
        Get all CSV files from the CIC DIAD 2024 dataset.

        Returns:
            List of CSV file paths
        """
        if not self.cicdiad2024_path:
            raise ValueError(
                "CIC DIAD 2024 dataset not downloaded yet. "
                "Call download_datasets() first."
            )

        csv_files = list(self.cicdiad2024_path.rglob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files in CIC DIAD 2024 dataset")
        return csv_files

    def prepare_data_directories(self) -> None:
        """Create necessary directories for data processing."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for organized data storage
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        (self.data_dir / "parquet").mkdir(exist_ok=True)

        logger.info(f"Data directories prepared at: {self.data_dir}")


# Global instance for easy access
extractor = DatasetExtractor()
