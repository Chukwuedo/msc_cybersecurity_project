import marimo

__generated_with = "0.16.2"
app = marimo.App(
    width="medium",
    app_title="UoL MSc CyberSecurity Project - Knowledge Distillation",
    auto_download=["html"],
    sql_output="native",
)


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
    from src.ml.data.extract_data import DatasetExtractor
    from src.ml.data.convert_data import DataConverter
    from src.ml.evaluation.dataset_analyzer import (
        DatasetAnalyzer,
        DatasetComparator,
        analyze_dataset_file,
    )
    from src.utils.reporting import DatasetReporter
    from src.config.settings import Settings

    return (
        DatasetExtractor,
        DataConverter,
        DatasetAnalyzer,
        DatasetComparator,
        analyze_dataset_file,
        DatasetReporter,
        Settings,
    )


@app.cell
def _(Settings):
    # Initialize settings
    settings = Settings()
    print("Project configuration loaded successfully")
    print(f"Data path: {settings.data_path}")
    print(f"Model save path: {settings.model_save_path}")
    return (settings,)


@app.cell
def _(mo):
    mo.md(r"""#### Dataset Download and Extraction""")
    return


@app.cell
def _(DatasetExtractor, DatasetReporter, settings):
    # Initialize extractor and download datasets
    extractor = DatasetExtractor(settings.data_path)

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

    return (extractor, ciciot2023_files, cicdiad2024_files)


@app.cell
def _(mo):
    mo.md(r"""#### Data Conversion to Parquet""")
    return


@app.cell
def _(DataConverter, DatasetReporter, settings, ciciot2023_files, cicdiad2024_files):
    # Initialize converter and convert datasets
    converter = DataConverter(settings.data_path)

    DatasetReporter.print_section_header("Converting to Parquet Format")

    # Convert CICIOT 2023
    ciciot2023_combined_path = None
    if ciciot2023_files:
        ciciot2023_combined_path = converter.convert_csv_to_parquet(
            ciciot2023_files, "ciciot2023_combined.parquet"
        )

    # Convert CIC DIAD 2024
    cicdiad2024_combined_path = None
    if cicdiad2024_files:
        cicdiad2024_combined_path = converter.convert_csv_to_parquet(
            cicdiad2024_files, "cicdiad2024_combined.parquet"
        )

    # Report conversion status
    DatasetReporter.print_conversion_status(
        "CICIOT 2023 Combined", ciciot2023_files, ciciot2023_combined_path
    )
    DatasetReporter.print_conversion_status(
        "CIC DIAD 2024 Combined", cicdiad2024_files, cicdiad2024_combined_path
    )

    return (converter, ciciot2023_combined_path, cicdiad2024_combined_path)


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
    return (ciciot2023_analyzer,)


@app.cell
def _(analyze_dataset_file, cicdiad2024_combined_path):
    # Analyze CIC DIAD 2024 dataset
    cicdiad2024_analyzer = analyze_dataset_file(
        cicdiad2024_combined_path, "CIC DIAD 2024 Dataset"
    )
    return (cicdiad2024_analyzer,)


@app.cell
def _(mo):
    mo.md(r"""#### Dataset Comparison""")
    return


@app.cell
def _(DatasetComparator, ciciot2023_combined_path, cicdiad2024_combined_path):
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

    return (comparator,)


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
