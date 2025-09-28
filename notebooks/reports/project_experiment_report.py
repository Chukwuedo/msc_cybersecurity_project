import marimo

__generated_with = "0.16.2"
app = marimo.App(
    width="medium",
    app_title="UoL MSc CyberSecurity Project - Knowledge Distillation",
    auto_download=["html"],
    sql_output="native",
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Project Experiment Report""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Data Sourcing and Analysis""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    **Data Sources:**
    * CICIOT 2023 - Training and Initial test data
    * CIC IOT DI-AD 2024 - Unseen Test data to validate robustness
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

    # Import our custom modules
    from src.knowledge_distillation_ensemble.ml.data.extract_data import (
        DatasetExtractor,
    )
    from src.knowledge_distillation_ensemble.ml.data.convert_data import DataConverter
    from src.ml.evaluation.dataset_analyzer import (
        DatasetAnalyzer,
        DatasetComparator,
        analyze_dataset_file,
    )
    from src.utils.reporting import DatasetReporter
    from src.knowledge_distillation_ensemble.config.settings import Settings

    return (
        DataConverter,
        DatasetComparator,
        DatasetExtractor,
        DatasetReporter,
        Settings,
        analyze_dataset_file,
    )


@app.cell
def _(Settings):
    # Initialize settings
    settings = Settings()
    print("Project configuration loaded successfully")
    print(f"Data path: {settings.data_path}")
    print(f"Model save path: {settings.model_save_path}")
    return


@app.cell
def _(mo):
    mo.md(r"""#### Dataset Download and Extraction""")
    return


@app.cell
def _(DatasetExtractor, DatasetReporter):
    # Initialize extractor and download datasets
    extractor = DatasetExtractor()

    DatasetReporter.print_section_header("Downloading Datasets")

    # Download datasets if needed
    extractor.download_datasets()

    # Get CSV file lists
    ciciot2023_files = extractor.get_ciciot2023_csv_files()
    cicdiad2024_files = extractor.get_cicdiad2024_csv_files()

    # Report file counts
    DatasetReporter.print_file_list(ciciot2023_files, "CICIOT 2023 Files", max_files=3)
    DatasetReporter.print_file_list(
        cicdiad2024_files, "CIC DIAD 2024 Files", max_files=3
    )

    return cicdiad2024_files, ciciot2023_files


@app.cell
def _(mo):
    mo.md(r"""#### Data Conversion to Parquet""")
    return


@app.cell
def _(DataConverter, DatasetReporter, cicdiad2024_files, ciciot2023_files):
    # Initialize converter and convert datasets
    converter = DataConverter()

    DatasetReporter.print_section_header("Converting to Parquet Format")

    # Convert CICIOT 2023
    ciciot2023_combined_path = None
    if ciciot2023_files:
        ciciot2023_combined_path = converter.convert_csv_to_parquet(
            ciciot2023_files, "ciciot2023_combined"
        )

    # Convert CIC DIAD 2024
    cicdiad2024_combined_path = None
    if cicdiad2024_files:
        cicdiad2024_combined_path = converter.convert_csv_to_parquet(
            cicdiad2024_files, "cicdiad2024_combined"
        )

    # Report conversion status
    DatasetReporter.print_conversion_status(
        "CICIOT 2023 Combined", ciciot2023_files, ciciot2023_combined_path
    )
    DatasetReporter.print_conversion_status(
        "CIC DIAD 2024 Combined", cicdiad2024_files, cicdiad2024_combined_path
    )

    return cicdiad2024_combined_path, ciciot2023_combined_path


@app.cell
def _(mo):
    mo.md(r"""#### Dataset Statistics and Analysis""")
    return


@app.cell
def _(analyze_dataset_file, ciciot2023_combined_path):
    # Analyze CICIOT 2023 dataset
    ciciot2023_analyzer = analyze_dataset_file(
        ciciot2023_combined_path, "CICIOT 2023 Dataset"
    )
    return


@app.cell
def _(analyze_dataset_file, cicdiad2024_combined_path):
    # Analyze CIC DIAD 2024 dataset
    cicdiad2024_analyzer = analyze_dataset_file(
        cicdiad2024_combined_path, "CIC DIAD 2024 Dataset"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""#### Dataset Comparison""")
    return


@app.cell
def _(DatasetComparator, cicdiad2024_combined_path, ciciot2023_combined_path):
    # Compare the two datasets
    if (
        ciciot2023_combined_path
        and ciciot2023_combined_path.exists()
        and cicdiad2024_combined_path
        and cicdiad2024_combined_path.exists()
    ):
        comparator = DatasetComparator(
            ciciot2023_combined_path,
            cicdiad2024_combined_path,
            "CICIOT 2023",
            "CIC DIAD 2024",
        )

        comparator.print_comparison_report()

    else:
        print("Cannot compare datasets - one or both files not available")
        comparator = None

    return


@app.cell
def _(mo):
    mo.md(r"""#### Complete Column Inventory""")
    return


@app.cell
def _(DatasetAnalyzer, cicdiad2024_combined_path, ciciot2023_combined_path):
    # Extract complete column lists from both datasets
    ciciot2023_columns = []
    cicdiad2024_columns = []

    if ciciot2023_combined_path and ciciot2023_combined_path.exists():
        print("=== CICIOT 2023 COMPLETE COLUMN LIST ===")
        analyzer_2023 = DatasetAnalyzer(ciciot2023_combined_path)
        ciciot2023_columns = analyzer_2023.get_basic_stats()["column_names"]

        print(f"Total columns: {len(ciciot2023_columns)}")
        print("All columns:")
        for col_idx1, col_name1 in enumerate(ciciot2023_columns, 1):
            print(f"  {col_idx1:2d}. {col_name1}")

    if cicdiad2024_combined_path and cicdiad2024_combined_path.exists():
        print("\n=== CIC DIAD 2024 COMPLETE COLUMN LIST ===")
        analyzer_2024 = DatasetAnalyzer(cicdiad2024_combined_path)
        cicdiad2024_columns = analyzer_2024.get_basic_stats()["column_names"]

        print(f"Total columns: {len(cicdiad2024_columns)}")
        print("All columns:")
        for col_idx2, col_name2 in enumerate(cicdiad2024_columns, 1):
            print(f"  {col_idx2:2d}. {col_name2}")

    return cicdiad2024_columns, ciciot2023_columns


@app.cell
def _(cicdiad2024_columns, ciciot2023_columns, mo):
    # Create markdown documentation of all columns for reference

    # Build CICIOT 2023 column list
    ciciot_md = (
        f"**CICIOT 2023 Dataset Columns ({len(ciciot2023_columns)} total):**\n\n"
    )
    for md_idx1, md_col1 in enumerate(ciciot2023_columns, 1):
        ciciot_md += f"{md_idx1}. `{md_col1}`\n"

    # Build CIC DIAD 2024 column list
    cicdiad_md = (
        f"\n**CIC DIAD 2024 Dataset Columns ({len(cicdiad2024_columns)} total):**\n\n"
    )
    for md_idx2, md_col2 in enumerate(cicdiad2024_columns, 1):
        cicdiad_md += f"{md_idx2}. `{md_col2}`\n"

    # Combine into single markdown
    complete_column_reference = ciciot_md + cicdiad_md

    mo.md(complete_column_reference)
    return complete_column_reference


@app.cell
def _(cicdiad2024_columns, ciciot2023_columns, mo):
    # Additional analysis for column overlap and differences

    # Find any potential column matches (case-insensitive or similar names)
    ciciot_lower = {
        norm_col.lower().replace(" ", "_").replace("-", "_"): norm_col
        for norm_col in ciciot2023_columns
    }
    cicdiad_lower = {
        norm_col.lower().replace(" ", "_").replace("-", "_"): norm_col
        for norm_col in cicdiad2024_columns
    }

    potential_matches = []
    for normalized, original_ciciot in ciciot_lower.items():
        if normalized in cicdiad_lower:
            potential_matches.append((original_ciciot, cicdiad_lower[normalized]))

    analysis_md = f"""
    #### Column Analysis Summary
    
    **Dataset Comparison:**
    - CICIOT 2023: {len(ciciot2023_columns)} columns
    - CIC DIAD 2024: {len(cicdiad2024_columns)} columns
    - Exact matches: 0 (as expected - different collection methodologies)
    - Potential semantic matches: {len(potential_matches)}
    
    """

    if potential_matches:
        analysis_md += "**Potential Semantic Matches:**\n\n"
        for ciciot_col, cicdiad_col in potential_matches:
            analysis_md += f"- `{ciciot_col}` ↔ `{cicdiad_col}`\n"

    analysis_md += """
    **Key Observations:**
    - CICIOT 2023 uses shorter, more abbreviated column names
    - CIC DIAD 2024 uses more descriptive, standardized network flow
      feature names
    - Both datasets capture network flow characteristics but with different
      feature engineering approaches
    - Feature mapping will be required for cross-dataset analysis and
      domain adaptation
    """

    mo.md(analysis_md)
    return analysis_md, potential_matches


@app.cell
def _(mo):
    mo.md(r"""### Machine Learning Pipeline""")
    return


@app.cell
def _(mo):
    mo.md(r"""#### Data Preprocessing""")
    return


@app.cell
def _():
    # TODO: Implement data preprocessing
    # - Feature scaling
    # - Encoding categorical variables
    # - Train/validation/test splits
    # - Data augmentation techniques
    print("Data preprocessing pipeline - To be implemented")
    return


@app.cell
def _(mo):
    mo.md(r"""#### Teacher Model Training""")
    return


@app.cell
def _():
    # TODO: Implement teacher model training
    # - Large, complex ensemble model
    # - Multiple algorithms (Random Forest, Gradient Boosting, Neural Network)
    # - Cross-validation and hyperparameter tuning
    # - Model evaluation and selection
    print("Teacher model training pipeline - To be implemented")
    return


@app.cell
def _(mo):
    mo.md(r"""#### Knowledge Distillation""")
    return


@app.cell
def _():
    # TODO: Implement knowledge distillation
    # - Student model architecture design
    # - Distillation loss function implementation
    # - Temperature scaling
    # - Training loop with teacher guidance
    print("Knowledge distillation pipeline - To be implemented")
    return


@app.cell
def _(mo):
    mo.md(r"""#### Student Model Evaluation""")
    return


@app.cell
def _():
    # TODO: Implement student model evaluation
    # - Performance metrics comparison
    # - Robustness testing on CIC DIAD 2024
    # - Model size and inference speed analysis
    # - Visualization of results
    print("Student model evaluation pipeline - To be implemented")
    return


@app.cell
def _(mo):
    mo.md(r"""### Results and Conclusions""")
    return


@app.cell
def _():
    # TODO: Implement results analysis
    # - Performance comparison tables
    # - Statistical significance tests
    # - Discussion of findings
    # - Future work recommendations
    print("Results analysis - To be implemented")
    return


if __name__ == "__main__":
    app.run()
