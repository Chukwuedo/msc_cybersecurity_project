"""Dataset analysis and statistics module for cybersecurity datasets.

This module provides comprehensive analysis capabilities for large datasets,
particularly focused on the CICIOT2023 and CICDIAD2024 cybersecurity datasets.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import polars as pl


class DatasetAnalyzer:
    """Analyzer for cybersecurity datasets with comprehensive statistics."""

    def __init__(self, dataset_path: Path):
        """Initialize analyzer with dataset path.

        Args:
            dataset_path: Path to the parquet dataset file
        """
        self.dataset_path = Path(dataset_path)
        self._dataset: Optional[pl.DataFrame] = None
        self._stats_cache: Dict[str, Any] = {}

    @property
    def dataset(self) -> pl.DataFrame:
        """Lazy load the dataset."""
        if self._dataset is None:
            if not self.dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
            self._dataset = pl.read_parquet(self.dataset_path)
        return self._dataset

    def get_basic_stats(self) -> Dict[str, Any]:
        """Get basic dataset statistics.

        Returns:
            Dictionary with shape, memory usage, and basic info
        """
        if "basic_stats" not in self._stats_cache:
            df = self.dataset
            stats = {
                "shape": df.shape,
                "rows": len(df),
                "columns": len(df.columns),
                "estimated_memory_mb": df.estimated_size("mb"),
                "column_names": df.columns,
            }
            self._stats_cache["basic_stats"] = stats
        return self._stats_cache["basic_stats"]

    def get_data_types_summary(self) -> Dict[str, int]:
        """Get summary of data types in the dataset.

        Returns:
            Dictionary mapping data type names to their counts
        """
        if "dtype_summary" not in self._stats_cache:
            dtypes = self.dataset.dtypes
            dtype_counts = {}
            for dtype in dtypes:
                dtype_str = str(dtype)
                dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1
            self._stats_cache["dtype_summary"] = dtype_counts
        return self._stats_cache["dtype_summary"]

    def get_missing_values_analysis(self) -> Dict[str, Any]:
        """Analyze missing values in the dataset.

        Returns:
            Dictionary with null counts, percentages, and problem columns
        """
        if "missing_analysis" not in self._stats_cache:
            df = self.dataset
            null_counts = df.null_count()
            null_values = null_counts.row(0)

            total_nulls = sum(null_values)
            total_cells = len(df) * len(df.columns)
            null_percentage = (
                (total_nulls / total_cells) * 100 if total_cells > 0 else 0
            )

            # Find columns with missing values
            columns_with_nulls = [
                col for col, count in zip(df.columns, null_values) if count > 0
            ]

            # Top columns with most nulls
            null_data = list(zip(df.columns, null_values))
            null_data_sorted = sorted(null_data, key=lambda x: x[1], reverse=True)

            top_null_columns = []
            for col, count in null_data_sorted[:10]:  # Top 10
                if count > 0:
                    pct = (count / len(df)) * 100
                    top_null_columns.append(
                        {"column": col, "null_count": count, "null_percentage": pct}
                    )

            analysis = {
                "total_nulls": total_nulls,
                "null_percentage": null_percentage,
                "columns_with_nulls_count": len(columns_with_nulls),
                "columns_with_nulls": columns_with_nulls,
                "top_null_columns": top_null_columns,
            }
            self._stats_cache["missing_analysis"] = analysis
        return self._stats_cache["missing_analysis"]

    def get_sample_data(self, n_rows: int = 5) -> pl.DataFrame:
        """Get a sample of the dataset.

        Args:
            n_rows: Number of rows to sample

        Returns:
            DataFrame with sampled data
        """
        return self.dataset.head(n_rows)

    def get_data_preview(self, n_rows: int = 100) -> Dict[str, Any]:
        """Get comprehensive data preview with sample and basic stats.

        Args:
            n_rows: Number of rows to include in preview

        Returns:
            Dictionary with sample data and basic analysis
        """
        df = self.dataset
        sample_data = df.head(n_rows)

        # Get basic statistics for numeric columns
        numeric_stats = {}
        try:
            numeric_cols = [
                col
                for col in df.columns
                if df[col].dtype in [pl.Float64, pl.Int64, pl.Float32, pl.Int32]
            ]
            if numeric_cols:
                stats_df = df.select(numeric_cols).describe()
                numeric_stats = stats_df.to_dict(as_series=False)
        except Exception:  # noqa: BLE001
            pass

        # Get value counts for categorical/string columns
        categorical_info = {}
        try:
            string_cols = [col for col in df.columns if df[col].dtype == pl.String][
                :5
            ]  # Limit to 5
            for col in string_cols:
                value_counts = df[col].value_counts().head(10)
                categorical_info[col] = value_counts.to_dict(as_series=False)
        except Exception:  # noqa: BLE001
            pass

        return {
            "sample_data": sample_data,
            "shape": df.shape,
            "numeric_stats": numeric_stats,
            "categorical_info": categorical_info,
            "column_dtypes": {
                col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)
            },
        }

    def get_column_preview(self, max_columns: int = 15) -> List[str]:
        """Get a preview of column names.

        Args:
            max_columns: Maximum number of columns to show

        Returns:
            List of column names (limited to max_columns)
        """
        columns = self.dataset.columns
        return columns[:max_columns]

    def print_comprehensive_report(self, dataset_name: str = "Dataset") -> None:
        """Print a comprehensive analysis report.

        Args:
            dataset_name: Name to use in the report header
        """
        print(f"=== {dataset_name.upper()} STATISTICS ===")

        # Basic statistics
        basic = self.get_basic_stats()
        print(f"Dataset shape: {basic['shape']}")
        print(f"Total rows: {basic['rows']:,}")
        print(f"Total columns: {basic['columns']}")
        print(f"Estimated memory usage: {basic['estimated_memory_mb']:.1f} MB")

        # Column preview
        print("\nColumn names (first 15):")
        columns = self.get_column_preview(15)
        for i, col in enumerate(columns, 1):
            print(f"  {i:2d}. {col}")
        if basic["columns"] > 15:
            remaining = basic["columns"] - 15
            print(f"  ... and {remaining} more columns")

        # Data types
        print("\nData types distribution:")
        dtypes = self.get_data_types_summary()
        for dtype, count in sorted(dtypes.items()):
            print(f"  {dtype}: {count} columns")

        # Missing values
        missing = self.get_missing_values_analysis()
        print("\nMissing values:")
        print(f"  Total null values: {missing['total_nulls']:,}")
        print(f"  Percentage of dataset: {missing['null_percentage']:.2f}%")
        print(f"  Columns with missing values: {missing['columns_with_nulls_count']}")

        if missing["top_null_columns"]:
            print("  Top 5 columns with most nulls:")
            for i, col_info in enumerate(missing["top_null_columns"][:5], 1):
                count = col_info["null_count"]
                pct = col_info["null_percentage"]
                print(f"    {i}. {col_info['column']}: {count:,} ({pct:.1f}%)")


class DatasetComparator:
    """Compare two datasets and analyze their differences."""

    def __init__(
        self,
        dataset1_path: Path,
        dataset2_path: Path,
        name1: str = "Dataset 1",
        name2: str = "Dataset 2",
    ):
        """Initialize comparator with two dataset paths.

        Args:
            dataset1_path: Path to first dataset
            dataset2_path: Path to second dataset
            name1: Name for first dataset
            name2: Name for second dataset
        """
        self.analyzer1 = DatasetAnalyzer(dataset1_path)
        self.analyzer2 = DatasetAnalyzer(dataset2_path)
        self.name1 = name1
        self.name2 = name2

    def compare_basic_stats(self) -> Dict[str, Any]:
        """Compare basic statistics between datasets."""
        stats1 = self.analyzer1.get_basic_stats()
        stats2 = self.analyzer2.get_basic_stats()

        return {
            "dataset1": {
                "name": self.name1,
                "rows": stats1["rows"],
                "columns": stats1["columns"],
                "memory_mb": stats1["estimated_memory_mb"],
            },
            "dataset2": {
                "name": self.name2,
                "rows": stats2["rows"],
                "columns": stats2["columns"],
                "memory_mb": stats2["estimated_memory_mb"],
            },
        }

    def find_common_columns(self) -> Tuple[set, set, set]:
        """Find common and unique columns between datasets.

        Returns:
            Tuple of (common_columns, unique_to_dataset1, unique_to_dataset2)
        """
        cols1 = set(self.analyzer1.get_basic_stats()["column_names"])
        cols2 = set(self.analyzer2.get_basic_stats()["column_names"])

        common = cols1 & cols2
        unique1 = cols1 - cols2
        unique2 = cols2 - cols1

        return common, unique1, unique2

    def print_comparison_report(self) -> None:
        """Print a comprehensive comparison report."""
        print("=== DATASET COMPARISON ANALYSIS ===")

        # Basic stats comparison
        comparison = self.compare_basic_stats()
        print(f"\n{self.name1}:")
        print(f"  Rows: {comparison['dataset1']['rows']:,}")
        print(f"  Columns: {comparison['dataset1']['columns']}")
        print(f"  Memory: {comparison['dataset1']['memory_mb']:.1f} MB")

        print(f"\n{self.name2}:")
        print(f"  Rows: {comparison['dataset2']['rows']:,}")
        print(f"  Columns: {comparison['dataset2']['columns']}")
        print(f"  Memory: {comparison['dataset2']['memory_mb']:.1f} MB")

        # Column analysis
        common, unique1, unique2 = self.find_common_columns()

        print("\nColumn Analysis:")
        print(f"  Common columns: {len(common)}")
        print(f"  Unique to {self.name1}: {len(unique1)}")
        print(f"  Unique to {self.name2}: {len(unique2)}")

        if common:
            print("  Common column examples (first 10):")
            for i, col in enumerate(sorted(common)[:10], 1):
                print(f"    {i}. {col}")

        if unique1:
            print(f"  Columns unique to {self.name1} (first 5):")
            for i, col in enumerate(sorted(unique1)[:5], 1):
                print(f"    {i}. {col}")

        if unique2:
            print(f"  Columns unique to {self.name2} (first 5):")
            for i, col in enumerate(sorted(unique2)[:5], 1):
                print(f"    {i}. {col}")


def analyze_dataset_file(
    dataset_path: Path, dataset_name: str
) -> Optional[DatasetAnalyzer]:
    """Convenience function to analyze a single dataset file.

    Args:
        dataset_path: Path to the dataset parquet file
        dataset_name: Name for the dataset in reports

    Returns:
        DatasetAnalyzer instance or None if file doesn't exist
    """
    if dataset_path and dataset_path.exists():
        try:
            analyzer = DatasetAnalyzer(dataset_path)
            analyzer.print_comprehensive_report(dataset_name)
            return analyzer
        except Exception as e:
            print(f"Error analyzing {dataset_name}: {e}")
            return None
    else:
        print(f"{dataset_name} not available at {dataset_path}")
        return None


def get_dataset_preview(dataset_path: Path, n_rows: int = 5) -> Optional[pl.DataFrame]:
    """Get a simple preview of a dataset as a DataFrame.

    Args:
        dataset_path: Path to the dataset parquet file
        n_rows: Number of rows to preview

    Returns:
        DataFrame with preview data or None if error
    """
    try:
        if dataset_path and dataset_path.exists():
            df = pl.read_parquet(dataset_path)
            return df.head(n_rows)
        else:
            print(f"Dataset not available at {dataset_path}")
            return None
    except Exception as e:
        print(f"Error creating preview: {e}")
        return None


def analyze_label_distributions(
    dataset_path: Path, label_column: str = "Label"
) -> Dict[str, Any]:
    """Analyze label value distributions in a dataset.

    Args:
        dataset_path: Path to the dataset parquet file
        label_column: Name of the label column (defaults to "Label")

    Returns:
        Dictionary with label statistics and distributions
    """
    try:
        if not dataset_path.exists():
            return {"error": f"Dataset not found at {dataset_path}"}

        # Use scan_parquet for memory efficiency with large datasets
        df = pl.scan_parquet(dataset_path)

        # Check if label column exists
        if label_column not in df.collect_schema().names():
            # Try common variations
            variations = ["Label", "label", "class", "Class", "target", "Target"]
            for var in variations:
                if var in df.collect_schema().names():
                    label_column = var
                    break
            else:
                return {"error": f"Label column not found in {dataset_path.name}"}

        # Get unique labels and their counts
        label_counts = (
            df.group_by(label_column).agg(pl.count().alias("count")).collect()
        )

        # Convert to dictionary for easier handling
        labels_dict = {}
        for row in label_counts.iter_rows(named=True):
            labels_dict[str(row[label_column])] = int(row["count"])

        # Sort by count in descending order
        sorted_labels = sorted(labels_dict.items(), key=lambda x: x[1], reverse=True)

        return {
            "unique_labels": [item[0] for item in sorted_labels],
            "label_counts": labels_dict,
            "total_records": sum(labels_dict.values()),
            "label_column": label_column,
        }
    except Exception as e:
        return {"error": f"Error analyzing labels: {e}"}


def compare_label_distributions(
    dataset1_path: Path,
    dataset2_path: Path,
    label_column1: str = "Label",
    label_column2: str = "Label",
) -> Dict[str, Any]:
    """Compare label distributions between two datasets.

    Args:
        dataset1_path: Path to first dataset
        dataset2_path: Path to second dataset
        label_column1: Label column name in first dataset
        label_column2: Label column name in second dataset

    Returns:
        Dictionary with comparison results
    """
    results1 = analyze_label_distributions(dataset1_path, label_column1)
    results2 = analyze_label_distributions(dataset2_path, label_column2)

    # Check for errors
    if "error" in results1 or "error" in results2:
        return {
            "dataset1": results1,
            "dataset2": results2,
            "error": "Error in one or both datasets",
        }

    # Get sets of unique labels
    labels1 = set(results1["unique_labels"])
    labels2 = set(results2["unique_labels"])

    # Find common and unique labels
    common_labels = labels1.intersection(labels2)
    unique_to_dataset1 = labels1.difference(labels2)
    unique_to_dataset2 = labels2.difference(labels1)

    # Group similar labels (case insensitive)
    lower_labels1 = {label.lower(): label for label in labels1}
    lower_labels2 = {label.lower(): label for label in labels2}

    similar_labels = {}
    for lower_label in lower_labels1:
        if (
            lower_label in lower_labels2
            and lower_labels1[lower_label] != lower_labels2[lower_label]
        ):
            similar_labels[lower_labels1[lower_label]] = lower_labels2[lower_label]

    # Calculate percentages for each dataset
    percentages1 = {
        label: (count / results1["total_records"]) * 100
        for label, count in results1["label_counts"].items()
    }

    percentages2 = {
        label: (count / results2["total_records"]) * 100
        for label, count in results2["label_counts"].items()
    }

    return {
        "dataset1": {
            "name": dataset1_path.name,
            "unique_labels": sorted(list(labels1)),
            "label_counts": results1["label_counts"],
            "label_percentages": percentages1,
            "total_records": results1["total_records"],
        },
        "dataset2": {
            "name": dataset2_path.name,
            "unique_labels": sorted(list(labels2)),
            "label_counts": results2["label_counts"],
            "label_percentages": percentages2,
            "total_records": results2["total_records"],
        },
        "common_labels": sorted(list(common_labels)),
        "unique_to_dataset1": sorted(list(unique_to_dataset1)),
        "unique_to_dataset2": sorted(list(unique_to_dataset2)),
        "similar_labels": similar_labels,
        "all_unique_labels": sorted(list(labels1.union(labels2))),
    }
