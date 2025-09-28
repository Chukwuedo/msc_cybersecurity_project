"""Data visualization utilities for cybersecurity dataset analysis.

This module provides clean, reusable visualization functions for
dataset analysis reports and notebooks.
"""

from typing import List, Dict, Any, Optional


class DatasetReporter:
    """Clean reporting utilities for dataset analysis."""

    @staticmethod
    def print_section_header(title: str, level: int = 1) -> None:
        """Print a formatted section header.

        Args:
            title: The section title
            level: Header level (1, 2, or 3)
        """
        if level == 1:
            print(f"=== {title.upper()} ===")
        elif level == 2:
            print(f"\n--- {title} ---")
        else:
            print(f"\n{title}:")

    @staticmethod
    def print_file_list(files: List, title: str, max_files: int = 3) -> None:
        """Print a formatted list of files.

        Args:
            files: List of file paths
            title: Title for the file list
            max_files: Maximum number of files to display
        """
        print(f"\n{title} ({len(files)} files):")
        if files:
            display_files = files[:max_files]
            for file_idx, file_path in enumerate(display_files, 1):
                print(f"  {file_idx}. {file_path.name}")
            if len(files) > max_files:
                remaining = len(files) - max_files
                print(f"  ... and {remaining} more files")
        else:
            print("  No files found")

    @staticmethod
    def print_status_check(condition: bool, success_msg: str, fail_msg: str) -> None:
        """Print a status check with checkmark or X.

        Args:
            condition: Whether the check passed
            success_msg: Message to show on success
            fail_msg: Message to show on failure
        """
        if condition:
            print(f"✓ {success_msg}")
        else:
            print(f"✗ {fail_msg}")

    @staticmethod
    def print_conversion_status(dataset_name: str, source_path, target_path) -> None:
        """Print dataset conversion status.

        Args:
            dataset_name: Name of the dataset
            source_path: Path to source files
            target_path: Path to target parquet file
        """
        source_exists = source_path and len(list(source_path)) > 0
        target_exists = target_path and target_path.exists()

        if source_exists and target_exists:
            print(f"✓ {dataset_name}: Ready")
        elif source_exists and not target_exists:
            print(f"⚠ {dataset_name}: Source found, needs conversion")
        elif not source_exists and target_exists:
            print(f"✓ {dataset_name}: Converted (source cleaned)")
        else:
            print(f"✗ {dataset_name}: Not available")

    @staticmethod
    def print_dataset_summary(
        name: str,
        shape: tuple,
        memory_mb: float,
        columns: List[str],
        max_cols: int = 15,
    ) -> None:
        """Print a clean dataset summary.

        Args:
            name: Dataset name
            shape: (rows, columns) tuple
            memory_mb: Memory usage in MB
            columns: List of column names
            max_cols: Maximum columns to display
        """
        print(f"\n{name} Summary:")
        print(f"  Shape: {shape[0]:,} rows × {shape[1]} columns")
        print(f"  Memory: {memory_mb:.1f} MB")

        if columns:
            print(f"  Columns (showing first {min(max_cols, len(columns))}):")
            for col_idx, col in enumerate(columns[:max_cols], 1):
                print(f"    {col_idx:2d}. {col}")
            if len(columns) > max_cols:
                remaining = len(columns) - max_cols
                print(f"    ... and {remaining} more")

    @staticmethod
    def print_data_preview(preview_data: dict, dataset_name: str) -> None:
        """Print a comprehensive data preview.

        Args:
            preview_data: Dictionary from DatasetAnalyzer.get_data_preview()
            dataset_name: Name of the dataset for the header
        """
        print(f"\n=== {dataset_name.upper()} DATA PREVIEW ===")

        sample_df = preview_data["sample_data"]
        shape = preview_data["shape"]

        print(f"Dataset Shape: {shape[0]:,} rows × {shape[1]} columns")
        print(f"Preview: First {len(sample_df)} rows")

        # Show sample data with better formatting
        print("\nSample Data:")
        print("-" * 80)

        # Display first few rows in a clean format
        for row_idx in range(min(5, len(sample_df))):
            print(f"Row {row_idx + 1}:")
            row_data = sample_df.row(row_idx, named=True)
            for col_name, value in row_data.items():
                # Truncate long values for readability
                str_val = str(value)
                if len(str_val) > 50:
                    str_val = str_val[:47] + "..."
                print(f"  {col_name}: {str_val}")
            print()

        # Show data types summary
        dtypes = preview_data["column_dtypes"]
        dtype_counts = {}
        for dtype in dtypes.values():
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1

        print("Data Types Summary:")
        for dtype, count in sorted(dtype_counts.items()):
            print(f"  {dtype}: {count} columns")

        # Show categorical info if available
        categorical_info = preview_data.get("categorical_info", {})
        if categorical_info:
            print("\nCategorical Columns (top values):")
            for col_name, value_counts in categorical_info.items():
                if "count" in value_counts and len(value_counts["count"]) > 0:
                    print(f"  {col_name}:")
                    values = value_counts.get("count", [])
                    categories = value_counts.get(col_name, [])
                    for i, (cat, count) in enumerate(zip(categories[:3], values[:3])):
                        print(f"    {cat}: {count}")
                    if len(categories) > 3:
                        print(f"    ... and {len(categories) - 3} more values")

        print("-" * 80)

    @staticmethod
    def print_comparison_table(
        dataset1_stats: Dict[str, Any], dataset2_stats: Dict[str, Any]
    ) -> None:
        """Print a side-by-side comparison table.

        Args:
            dataset1_stats: Stats dict for first dataset
            dataset2_stats: Stats dict for second dataset
        """
        name1 = dataset1_stats["name"]
        name2 = dataset2_stats["name"]

        # Calculate column widths
        name_width = max(len(name1), len(name2)) + 2

        print(f"\n{'Metric':<15} {'Dataset 1':<{name_width}} {'Dataset 2'}")
        print("-" * (15 + name_width + 15))
        print(f"{'Name':<15} {name1:<{name_width}} {name2}")
        print(
            f"{'Rows':<15} "
            f"{dataset1_stats['rows']:,}<{name_width} "
            f"{dataset2_stats['rows']:,}"
        )
        print(
            f"{'Columns':<15} "
            f"{dataset1_stats['columns']:<{name_width}} "
            f"{dataset2_stats['columns']}"
        )
        print(
            f"{'Memory (MB)':<15} "
            f"{dataset1_stats['memory_mb']:.1f}<{name_width} "
            f"{dataset2_stats['memory_mb']:.1f}"
        )


class SimpleProgressReporter:
    """Simple progress reporting for long operations."""

    def __init__(self, total_steps: int, operation_name: str = "Processing"):
        """Initialize progress reporter.

        Args:
            total_steps: Total number of steps
            operation_name: Name of the operation
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.operation_name = operation_name

    def step(self, description: str = "") -> None:
        """Advance one step and print progress.

        Args:
            description: Optional description of current step
        """
        self.current_step += 1
        percentage = (self.current_step / self.total_steps) * 100

        status = f"[{self.current_step}/{self.total_steps}] "
        status += f"({percentage:.0f}%) {self.operation_name}"
        if description:
            status += f": {description}"

        print(status)

    def finish(self, success_message: str = "Completed") -> None:
        """Mark operation as finished.

        Args:
            success_message: Message to display on completion
        """
        print(f"✓ {self.operation_name}: {success_message}")


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string (e.g., "1.5 MB", "3.2 GB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def format_number(num: int) -> str:
    """Format large numbers with commas.

    Args:
        num: Number to format

    Returns:
        Formatted number string (e.g., "1,234,567")
    """
    return f"{num:,}"


def create_simple_table(
    headers: List[str], rows: List[List[str]], title: Optional[str] = None
) -> None:
    """Create a simple ASCII table.

    Args:
        headers: List of column headers
        rows: List of row data (each row is a list of strings)
        title: Optional table title
    """
    if title:
        print(f"\n{title}")

    # Calculate column widths
    col_widths = [len(header) for header in headers]
    for row in rows:
        for col_idx, cell in enumerate(row):
            if col_idx < len(col_widths):
                col_widths[col_idx] = max(col_widths[col_idx], len(str(cell)))

    # Print header
    header_line = " | ".join(
        header.ljust(col_widths[col_idx]) for col_idx, header in enumerate(headers)
    )
    print(header_line)
    print("-" * len(header_line))

    # Print rows
    for row in rows:
        row_line = " | ".join(
            str(cell).ljust(col_widths[col_idx]) for col_idx, cell in enumerate(row)
        )
        print(row_line)
