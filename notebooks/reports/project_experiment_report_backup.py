import marimo

__generated_with = "0.16.2"
app = marimo.App(
    width="medium",
    app_title="UoL MSc CyberSecurity Project - Knowledge Distillation Ensemble",
    auto_download=["html"],
    sql_output="native",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Project Experiment Report""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Data Sourcing""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The two data sources to be used for this project are as follows:

    * CICIOT 2023 - Training and Initial test data
    * CIC IOT DI-AD 2024 - Unseen Test data to validate robustness of approach
    """
    )
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    # Add project root to Python path
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return


@app.cell
def _(mo):
    mo.md(r"""#### Data Schema Analysis""")
    return


@app.cell
def _(mo):
    mo.md(r"""#### Data Sample Exploration""")
    return


@app.cell
def _():
    from src.knowledge_distillation_ensemble.ml.data import extractor, converter

    return (extractor, converter)


@app.cell
def _(extractor):
    # Download both datasets and prepare directories
    extractor.prepare_data_directories()
    dataset_paths = extractor.download_datasets()

    print("Dataset download completed!")
    print(f"CICIOT 2023 path: {dataset_paths['ciciot2023']}")
    print(f"CIC DIAD 2024 path: {dataset_paths['cicdiad2024']}")
    return (dataset_paths,)


@app.cell
def _(extractor):
    # Get CSV files from both datasets
    ciciot2023_files = extractor.get_ciciot2023_csv_files()
    cicdiad2024_files = extractor.get_cicdiad2024_csv_files()

    print(f"CICIOT 2023 dataset contains {len(ciciot2023_files)} CSV files")
    print(f"CIC DIAD 2024 dataset contains {len(cicdiad2024_files)} CSV files")

    # Show first few files from each dataset
    print("\nFirst 3 CICIOT 2023 files:")
    for idx1, file in enumerate(ciciot2023_files[:3]):
        print(f"  {idx1 + 1}. {file.name}")

    print("\nFirst 3 CIC DIAD 2024 files:")
    for idx2, file in enumerate(cicdiad2024_files[:3]):
        print(f"  {idx2 + 1}. {file.name}")

    return (ciciot2023_files, cicdiad2024_files)


@app.cell
def _(mo):
    mo.md(r"""#### Data Quality Assessment""")
    return


@app.cell
def _(mo):
    mo.md(r"""#### Convert CSV to Parquet Format""")
    return


@app.cell
def _():
    # Import column selectors for data analysis
    import polars.selectors as cs

    return (cs,)


@app.cell
def _(mo):
    mo.md(r"""### Data Preprocessing""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Machine Learning Modelling""")
    return


@app.cell
def _(mo):
    mo.md(r"""#### Teacher Model Training""")
    return


@app.cell
def _(mo):
    mo.md(r"""#### Teacher Model Testing""")
    return


@app.cell
def _(mo):
    mo.md(r"""#### Knowledge Distillation of Student Model""")
    return


@app.cell
def _(mo):
    mo.md(r"""#### Student Model Testing""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Model Evaluations""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Conclusions""")
    return


@app.cell
def _(ciciot2023_files, converter):
    # Analyze schema of the first CICIOT 2023 file
    if ciciot2023_files:
        first_file = ciciot2023_files[0]
        print(f"Analyzing CICIOT 2023 schema: {first_file.name}")

        try:
            schema = converter.get_csv_schema(first_file)
            print(f"\nSchema contains {len(schema)} columns:")
            print(schema.head(10))  # Show first 10 columns
        except Exception as e:
            print(f"Error analyzing schema: {e}")

    return


@app.cell
def _(cicdiad2024_files, converter):
    # Analyze schema of the first CIC DIAD 2024 file
    if cicdiad2024_files:
        first_file_2024 = cicdiad2024_files[0]
        print(f"Analyzing CIC DIAD 2024 schema: {first_file_2024.name}")

        try:
            schema_2024 = converter.get_csv_schema(first_file_2024)
            print(f"\nSchema contains {len(schema_2024)} columns:")
            print(schema_2024.head(10))  # Show first 10 columns
        except Exception as e:
            print(f"Error analyzing schema: {e}")

    return (schema_2024,) if cicdiad2024_files else (None,)


@app.cell
def _(ciciot2023_files, converter):
    # Sample data from the first CICIOT 2023 file for exploration
    if ciciot2023_files:
        sample_file = ciciot2023_files[0]
        print(f"Sampling CICIOT 2023 data from: {sample_file.name}")

        # Sample 1000 rows for quick exploration
        sample_data = converter.sample_csv_data(sample_file, sample_size=1000)

        print(f"Sample data shape: {sample_data.shape}")
        print("\nColumn names:")
        print(sample_data.columns[:10])  # Show first 10 columns

        print("\nFirst few rows:")
        print(sample_data.head())

    return (sample_data,)


@app.cell
def _(cicdiad2024_files, converter):
    # Sample data from the first CIC DIAD 2024 file for exploration
    if cicdiad2024_files:
        sample_file_2024 = cicdiad2024_files[0]
        print(f"Sampling CIC DIAD 2024 data from: {sample_file_2024.name}")

        # Sample 1000 rows for quick exploration
        sample_data_2024 = converter.sample_csv_data(sample_file_2024, sample_size=1000)

        print(f"Sample data shape: {sample_data_2024.shape}")
        print("\nColumn names:")
        print(sample_data_2024.columns[:10])  # Show first 10 columns

        print("\nFirst few rows:")
        print(sample_data_2024.head())

    return (sample_data_2024,) if cicdiad2024_files else (None,)


@app.cell
def _(mo):
    mo.md(
        """
    **Converting datasets to Parquet format for efficient processing:**

    - Parquet format provides better compression and faster query performance
    - Enables efficient columnar operations for machine learning
    """
    )
    return


@app.cell
def _(ciciot2023_files, converter):
    # Convert a subset of CICIOT 2023 files to Parquet for testing
    if ciciot2023_files:
        # Convert first file as a test
        test_file = ciciot2023_files[0]
        print(f"Converting CICIOT 2023 {test_file.name} to Parquet...")

        try:
            # Convert with a sample for testing
            # (remove sample_rows for full data)
            parquet_path = converter.convert_csv_to_parquet(
                csv_paths=test_file,
                output_name="ciciot2023_sample",
                sample_rows=5000,  # Sample for testing
            )

            print("CICIOT 2023 sample conversion completed!")
            print(f"Parquet file saved to: {parquet_path}")

        except Exception as e:
            print(f"Conversion failed: {e}")
            parquet_path = None

    return (parquet_path,) if ciciot2023_files else (None,)


@app.cell
def _(cicdiad2024_files, converter):
    # Convert a subset of CIC DIAD 2024 files to Parquet for testing
    if cicdiad2024_files:
        # Convert first file as a test
        test_file_2024 = cicdiad2024_files[0]
        print(f"Converting CIC DIAD 2024 {test_file_2024.name} to Parquet...")

        try:
            # Convert with a sample for testing
            # (remove sample_rows for full data)
            parquet_path_2024 = converter.convert_csv_to_parquet(
                csv_paths=test_file_2024,
                output_name="cicdiad2024_sample",
                sample_rows=5000,  # Sample for testing
            )

            print("CIC DIAD 2024 sample conversion completed!")
            print(f"Parquet file saved to: {parquet_path_2024}")

        except Exception as e:
            print(f"Conversion failed: {e}")
            parquet_path_2024 = None

    return (parquet_path_2024,) if cicdiad2024_files else (None,)


@app.cell
def _(mo):
    mo.md(r"""#### Combine All CICIOT 2023 CSV Files""")
    return


@app.cell
def _(ciciot2023_files, converter):
    # Combine all CICIOT 2023 CSV files into a single Parquet file
    if ciciot2023_files:
        print(f"Combining {len(ciciot2023_files)} CICIOT 2023 CSV files...")
        print("This may take several minutes depending on data size...")

        try:
            # Convert all CSV files to a single Parquet file
            # Remove sample_rows parameter to process all data
            ciciot2023_combined_path = converter.convert_csv_to_parquet(
                csv_paths=ciciot2023_files,
                output_name="ciciot2023_combined",
                # sample_rows=None  # Process all data
            )

            print("CICIOT 2023 combination completed!")
            print(f"Combined Parquet file: {ciciot2023_combined_path}")

        except Exception as e:
            print(f"CICIOT 2023 combination failed: {e}")
            ciciot2023_combined_path = None
    else:
        ciciot2023_combined_path = None

    return (ciciot2023_combined_path,)


@app.cell
def _(mo):
    mo.md(r"""#### Combine All CIC DIAD 2024 CSV Files""")
    return


@app.cell
def _(cicdiad2024_files, converter):
    # Combine all CIC DIAD 2024 CSV files into a single Parquet file
    if cicdiad2024_files:
        print(f"Combining {len(cicdiad2024_files)} CIC DIAD 2024 CSV files...")
        print("This may take several minutes depending on data size...")

        try:
            # Convert all CSV files to a single Parquet file
            # Remove sample_rows parameter to process all data
            cicdiad2024_combined_path = converter.convert_csv_to_parquet(
                csv_paths=cicdiad2024_files,
                output_name="cicdiad2024_combined",
                # sample_rows=None  # Process all data
            )

            print("CIC DIAD 2024 combination completed!")
            print(f"Combined Parquet file: {cicdiad2024_combined_path}")

        except Exception as e:
            print(f"CIC DIAD 2024 combination failed: {e}")
            cicdiad2024_combined_path = None
    else:
        cicdiad2024_combined_path = None

    return (cicdiad2024_combined_path,)


@app.cell
def _(mo):
    mo.md(r"""#### Combined Dataset Summary""")
    return


@app.cell
def _(ciciot2023_combined_path, cicdiad2024_combined_path):
    # Summary of combined datasets
    print("=== COMBINED DATASET SUMMARY ===")

    if ciciot2023_combined_path:
        print(f"✓ CICIOT 2023 Combined: {ciciot2023_combined_path}")
        size_mb = ciciot2023_combined_path.stat().st_size / (1024 * 1024)
        print(f"  File size: {size_mb:.1f} MB")
    else:
        print("✗ CICIOT 2023 Combined: Failed")

    if cicdiad2024_combined_path:
        print(f"✓ CIC DIAD 2024 Combined: {cicdiad2024_combined_path}")
        size_mb = cicdiad2024_combined_path.stat().st_size / (1024 * 1024)
        print(f"  File size: {size_mb:.1f} MB")
    else:
        print("✗ CIC DIAD 2024 Combined: Failed")

    return


@app.cell
def _(mo):
    mo.md(r"""#### Full Dataset Statistics Analysis""")
    return


@app.cell
def _(ciciot2023_combined_path):
    # Load and analyze the full CICIOT 2023 combined dataset
    if ciciot2023_combined_path and ciciot2023_combined_path.exists():
        print("=== FULL CICIOT 2023 DATASET STATISTICS ===")
        
        try:
            # Load the full Parquet file
            import polars as pl
            
            print("Loading CICIOT 2023 combined dataset...")
            ciciot2023_full = pl.read_parquet(ciciot2023_combined_path)
            
            # Basic statistics
            print(f"Dataset shape: {ciciot2023_full.shape}")
            print(f"Total rows: {len(ciciot2023_full):,}")
            print(f"Total columns: {len(ciciot2023_full.columns)}")
            
            # Memory usage
            estimated_memory = ciciot2023_full.estimated_size("mb")
            print(f"Estimated memory usage: {estimated_memory:.1f} MB")
            
            # Column information
            print("\nColumn names (first 15):")
            for idx3, col in enumerate(ciciot2023_full.columns[:15]):
                print(f"  {idx3+1:2d}. {col}")
            if len(ciciot2023_full.columns) > 15:
                remaining = len(ciciot2023_full.columns) - 15
                print(f"  ... and {remaining} more columns")
                
            # Data types summary
            dtypes = ciciot2023_full.dtypes
            dtype_counts = {}
            for dtype in dtypes:
                dtype_str = str(dtype)
                dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1
            
            print("\nData types distribution:")
            for dtype, count in sorted(dtype_counts.items()):
                print(f"  {dtype}: {count} columns")
            
            # Missing values analysis
            null_counts = ciciot2023_full.null_count()
            total_nulls = sum(null_counts.row(0))
            print("\nMissing values:")
            print(f"  Total null values: {total_nulls:,}")
            
            total_cells = len(ciciot2023_full) * len(ciciot2023_full.columns)
            null_percentage = (total_nulls / total_cells) * 100
            print(f"  Percentage of dataset: {null_percentage:.2f}%")
            
            # Columns with missing values
            columns_with_nulls = [
                col for col, count in zip(
                    ciciot2023_full.columns, null_counts.row(0)
                )
                if count > 0
            ]
            print(f"  Columns with missing values: {len(columns_with_nulls)}")
            
            if columns_with_nulls:
                print("  Top 5 columns with most nulls:")
                null_data = list(zip(
                    ciciot2023_full.columns, null_counts.row(0)
                ))
                null_data_sorted = sorted(
                    null_data, key=lambda x: x[1], reverse=True
                )
                for idx4, (col, count) in enumerate(null_data_sorted[:5]):
                    if count > 0:
                        pct = (count / len(ciciot2023_full)) * 100
                        print(f"    {idx4+1}. {col}: {count:,} ({pct:.1f}%)")
            
        except Exception as e:
            print(f"Error analyzing CICIOT 2023 dataset: {e}")
            ciciot2023_full = None
    else:
        print("CICIOT 2023 combined dataset not available")
        ciciot2023_full = None
        
    return (ciciot2023_full,)


@app.cell
def _(cicdiad2024_combined_path):
    # Load and analyze the full CIC DIAD 2024 combined dataset
    if cicdiad2024_combined_path and cicdiad2024_combined_path.exists():
        print("=== FULL CIC DIAD 2024 DATASET STATISTICS ===")
        
        try:
            # Load the full Parquet file
            import polars as pl
            
            print("Loading CIC DIAD 2024 combined dataset...")
            cicdiad2024_full = pl.read_parquet(cicdiad2024_combined_path)
            
            # Basic statistics
            print(f"Dataset shape: {cicdiad2024_full.shape}")
            print(f"Total rows: {len(cicdiad2024_full):,}")
            print(f"Total columns: {len(cicdiad2024_full.columns)}")
            
            # Memory usage
            estimated_memory = cicdiad2024_full.estimated_size("mb")
            print(f"Estimated memory usage: {estimated_memory:.1f} MB")
            
            # Column information
            print("\nColumn names (first 15):")
            for idx5, col in enumerate(cicdiad2024_full.columns[:15]):
                print(f"  {idx5+1:2d}. {col}")
            if len(cicdiad2024_full.columns) > 15:
                remaining = len(cicdiad2024_full.columns) - 15
                print(f"  ... and {remaining} more columns")
                
            # Data types summary
            dtypes = cicdiad2024_full.dtypes
            dtype_counts = {}
            for dtype in dtypes:
                dtype_str = str(dtype)
                dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1
            
            print("\nData types distribution:")
            for dtype, count in sorted(dtype_counts.items()):
                print(f"  {dtype}: {count} columns")
            
            # Missing values analysis
            null_counts = cicdiad2024_full.null_count()
            total_nulls = sum(null_counts.row(0))
            print("\nMissing values:")
            print(f"  Total null values: {total_nulls:,}")
            
            total_cells = len(cicdiad2024_full) * len(cicdiad2024_full.columns)
            null_percentage = (total_nulls / total_cells) * 100
            print(f"  Percentage of dataset: {null_percentage:.2f}%")
            
            # Columns with missing values
            columns_with_nulls = [
                col for col, count in zip(
                    cicdiad2024_full.columns, null_counts.row(0)
                )
                if count > 0
            ]
            print(f"  Columns with missing values: {len(columns_with_nulls)}")
            
            if columns_with_nulls:
                print("  Top 5 columns with most nulls:")
                null_data = list(zip(
                    cicdiad2024_full.columns, null_counts.row(0)
                ))
                null_data_sorted = sorted(
                    null_data, key=lambda x: x[1], reverse=True
                )
                for idx6, (col, count) in enumerate(null_data_sorted[:5]):
                    if count > 0:
                        pct = (count / len(cicdiad2024_full)) * 100
                        print(f"    {idx6+1}. {col}: {count:,} ({pct:.1f}%)")
            
        except Exception as e:
            print(f"Error analyzing CIC DIAD 2024 dataset: {e}")
            cicdiad2024_full = None
    else:
        print("CIC DIAD 2024 combined dataset not available")
        cicdiad2024_full = None
        
    return (cicdiad2024_full,)


@app.cell
def _(ciciot2023_full, cicdiad2024_full):
    # Comparative analysis between the two datasets
    print("=== DATASET COMPARISON ===")
    
    if ciciot2023_full is not None and cicdiad2024_full is not None:
        print("\n📊 Size Comparison:")
        print(f"CICIOT 2023 rows:    {len(ciciot2023_full):,}")
        print(f"CIC DIAD 2024 rows:  {len(cicdiad2024_full):,}")
        
        ratio = len(ciciot2023_full) / len(cicdiad2024_full)
        print(f"Size ratio (2023/2024): {ratio:.2f}x")
        
        print("\n📋 Column Comparison:")
        print(f"CICIOT 2023 columns:    {len(ciciot2023_full.columns)}")
        print(f"CIC DIAD 2024 columns:  {len(cicdiad2024_full.columns)}")
        
        # Check for common columns
        common_cols = (
            set(ciciot2023_full.columns) & set(cicdiad2024_full.columns)
        )
        print(f"Common columns: {len(common_cols)}")
        
        if common_cols:
            print("Common column examples (first 10):")
            for idx7, col in enumerate(sorted(common_cols)[:10]):
                print(f"  {idx7+1}. {col}")
        
        # Unique columns
        unique_2023 = (
            set(ciciot2023_full.columns) - set(cicdiad2024_full.columns)
        )
        unique_2024 = (
            set(cicdiad2024_full.columns) - set(ciciot2023_full.columns)
        )
        
        print(f"\nUnique to CICIOT 2023: {len(unique_2023)} columns")
        print(f"Unique to CIC DIAD 2024: {len(unique_2024)} columns")
        
        print("\n💾 Storage Efficiency:")
        if hasattr(ciciot2023_full, 'estimated_size'):
            mem_2023 = ciciot2023_full.estimated_size("mb")
            mem_2024 = cicdiad2024_full.estimated_size("mb")
            print(f"CICIOT 2023 memory:     {mem_2023:.1f} MB")
            print(f"CIC DIAD 2024 memory:   {mem_2024:.1f} MB")
            print(f"Total memory usage:     {mem_2023 + mem_2024:.1f} MB")
        
    elif ciciot2023_full is not None:
        print("✓ CICIOT 2023 dataset loaded successfully")
        print("✗ CIC DIAD 2024 dataset not available for comparison")
    elif cicdiad2024_full is not None:
        print("✗ CICIOT 2023 dataset not available for comparison")
        print("✓ CIC DIAD 2024 dataset loaded successfully")
    else:
        print("✗ Neither dataset available for analysis")
        
    return


@app.cell
def _(sample_data):
    # Analyze CICIOT 2023 data quality if sample data is available
    if sample_data is not None:
        print("=== CICIOT 2023 Data Quality Analysis ===")
        print(f"Total rows: {len(sample_data)}")
        print(f"Total columns: {len(sample_data.columns)}")

        # Check for missing values
        null_counts = sample_data.null_count()
        columns_with_nulls = [
            col
            for col, count in zip(sample_data.columns, null_counts.row(0))
            if count > 0
        ]

        print(f"Columns with missing values: {len(columns_with_nulls)}")
        if columns_with_nulls:
            print("Columns with nulls:", columns_with_nulls[:5])

        # Show data types
        print("\nData types summary:")
        print(f"Total columns: {len(sample_data.columns)}")

    return


@app.cell
def _(sample_data_2024):
    # Analyze CIC DIAD 2024 data quality if sample data is available
    if sample_data_2024 is not None:
        print("=== CIC DIAD 2024 Data Quality Analysis ===")
        print(f"Total rows: {len(sample_data_2024)}")
        print(f"Total columns: {len(sample_data_2024.columns)}")

        # Check for missing values
        null_counts_2024 = sample_data_2024.null_count()
        columns_with_nulls_2024 = [
            col
            for col, count in zip(sample_data_2024.columns, null_counts_2024.row(0))
            if count > 0
        ]

        print(f"Columns with missing values: {len(columns_with_nulls_2024)}")
        if columns_with_nulls_2024:
            print("Columns with nulls:", columns_with_nulls_2024[:5])

        # Show data types
        print("\nData types summary:")
        print(f"Total columns: {len(sample_data_2024.columns)}")

    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
