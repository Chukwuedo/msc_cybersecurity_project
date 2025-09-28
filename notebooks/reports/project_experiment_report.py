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
    import marimo as mo
    return (mo,)


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
    from src.knowledge_distillation_ensemble.ml.data.dataset_analyzer import (
        DatasetAnalyzer,
        DatasetComparator,
        analyze_dataset_file,
        get_dataset_preview,
    )
    from src.knowledge_distillation_ensemble.config.settings import Settings
    return (
        DataConverter,
        DatasetAnalyzer,
        DatasetComparator,
        DatasetExtractor,
        Settings,
        analyze_dataset_file,
        get_dataset_preview,
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
def _(DatasetExtractor):
    # Initialize extractor and download datasets
    extractor = DatasetExtractor()

    print("=== DOWNLOADING DATASETS ===")

    # Download datasets if needed
    extractor.download_datasets()

    # Get CSV file lists
    ciciot2023_files = extractor.get_ciciot2023_csv_files()
    cicdiad2024_files = extractor.get_cicdiad2024_csv_files()

    # Report file counts
    print(f"\nCICIOT 2023 Files ({len(ciciot2023_files)} files):")
    if ciciot2023_files:
        for idx, file_path in enumerate(ciciot2023_files[:3], 1):
            print(f"  {idx}. {file_path.name}")
        if len(ciciot2023_files) > 3:
            remaining = len(ciciot2023_files) - 3
            print(f"  ... and {remaining} more files")
    else:
        print("  No files found")

    print(f"\nCIC DIAD 2024 Files ({len(cicdiad2024_files)} files):")
    if cicdiad2024_files:
        for idx, file_path in enumerate(cicdiad2024_files[:3], 1):
            print(f"  {idx}. {file_path.name}")
        if len(cicdiad2024_files) > 3:
            remaining = len(cicdiad2024_files) - 3
            print(f"  ... and {remaining} more files")
    else:
        print("  No files found")
    return cicdiad2024_files, ciciot2023_files


@app.cell
def _(mo):
    mo.md(r"""#### Data Conversion to Parquet""")
    return


@app.cell
def _(DataConverter, cicdiad2024_files, ciciot2023_files):
    # Initialize converter and convert datasets
    converter = DataConverter()

    print("=== CONVERTING TO PARQUET FORMAT ===")

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
    def print_conversion_status(name, source_files, target_path):
        source_exists = source_files and len(list(source_files)) > 0
        target_exists = target_path and target_path.exists()

        if source_exists and target_exists:
            print(f"✓ {name}: Ready")
        elif source_exists and not target_exists:
            print(f"⚠ {name}: Source found, needs conversion")
        elif not source_exists and target_exists:
            print(f"✓ {name}: Converted (source cleaned)")
        else:
            print(f"✗ {name}: Not available")

    print_conversion_status(
        "CICIOT 2023 Combined", ciciot2023_files, ciciot2023_combined_path
    )
    print_conversion_status(
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
    return


@app.cell
def _(mo):
    mo.md(r"""#### Complete Feature Mapping Table""")
    return


@app.cell
def _(cicdiad2024_columns, ciciot2023_columns, mo):
    # Feature mapping analysis based on manual review

    # Define the common feature mapping table
    feature_mapping_data = [
        (
            "flow_duration",
            "['flow_duration']",
            "['Flow Duration']",
            "direct",
            "Same concept in both datasets: total duration of the flow/window.",
        ),
        (
            "flow_packets_per_second",
            "['Rate']",
            "['Flow Packets/s']",
            "direct",
            "Packets per second over the whole flow.",
        ),
        (
            "forward_packets_per_second",
            "['Srate']",
            "['Fwd Packets/s']",
            "direct",
            "Forward-direction packet rate.",
        ),
        (
            "backward_packets_per_second",
            "['Drate']",
            "['Bwd Packets/s']",
            "direct",
            "Backward-direction packet rate.",
        ),
        (
            "flow_iat_mean",
            "['IAT']",
            "['Flow IAT Mean']",
            "direct",
            "Mean inter-arrival time across packets in the flow.",
        ),
        (
            "packet_length_min",
            "['Min']",
            "['Packet Length Min']",
            "direct",
            "Minimum packet length observed in the flow.",
        ),
        (
            "packet_length_max",
            "['Max']",
            "['Packet Length Max']",
            "direct",
            "Maximum packet length observed in the flow.",
        ),
        (
            "packet_length_mean",
            "['AVG']",
            "['Packet Length Mean']",
            "direct",
            "Average packet length across the flow.",
        ),
        (
            "packet_length_std",
            "['Std']",
            "['Packet Length Std']",
            "direct",
            "Standard deviation of packet length across the flow.",
        ),
        (
            "packet_length_range",
            "['Max - Min']",
            "['Packet Length Max - Packet Length Min']",
            "engineered",
            "Range computed as max - min in both datasets.",
        ),
        (
            "fin_flag_count",
            "['fin_flag_number', 'fin_count']",
            "['FIN Flag Count']",
            "direct",
            "FIN occurrences across the flow.",
        ),
        (
            "syn_flag_count",
            "['syn_flag_number', 'syn_count']",
            "['SYN Flag Count']",
            "direct",
            "SYN occurrences across the flow.",
        ),
        (
            "rst_flag_count",
            "['rst_flag_number', 'rst_count']",
            "['RST Flag Count']",
            "direct",
            "RST occurrences across the flow.",
        ),
        (
            "psh_flag_count",
            "['psh_flag_number']",
            "['PSH Flag Count']",
            "direct",
            "PSH occurrences across the flow.",
        ),
        (
            "ack_flag_count",
            "['ack_flag_number', 'ack_count']",
            "['ACK Flag Count']",
            "direct",
            "ACK occurrences across the flow.",
        ),
        (
            "ece_flag_count",
            "['ece_flag_number']",
            "['ECE Flag Count']",
            "direct",
            "ECE occurrences across the flow.",
        ),
        (
            "cwr_flag_count",
            "['cwr_flag_number']",
            "['CWR Flag Count']",
            "direct",
            "CWR occurrences across the flow.",
        ),
        (
            "urg_flag_count",
            "['urg_count']",
            "['URG Flag Count']",
            "direct",
            "URG occurrences across the flow.",
        ),
        (
            "total_packets",
            "['Number']",
            "['Total Fwd Packet + Total Bwd packets']",
            "composite_sum",
            "Total packets over both directions.",
        ),
        (
            "total_bytes",
            "['Tot sum']",
            "['Total Length of Fwd Packet + Total Length of Bwd Packet']",
            "composite_sum",
            "Total bytes over both directions.",
        ),
        (
            "average_packet_size",
            "['Tot size']",
            "['Average Packet Size']",
            "direct_or_equivalent",
            "Mean packet size over the flow.",
        ),
        (
            "flow_bytes_per_second",
            "['Tot sum / flow_duration']",
            "['Flow Bytes/s']",
            "engineered",
            "Normalised byte rate; compute from total bytes and duration.",
        ),
        (
            "header_length_total",
            "['Header_Length']",
            "['Fwd Header Length + Bwd Header Length']",
            "composite_sum",
            "Total transport/network header length across directions.",
        ),
        (
            "label",
            "['label']",
            "['Label']",
            "direct",
            "Target variable. Will harmonise this at a later stage.",
        ),
    ]

    analysis_md = f"""
    #### Feature Mapping Analysis

    **Dataset Overview:**

    - **CICIOT 2023**: {len(ciciot2023_columns)} columns (abbreviated naming)
    - **CIC DIAD 2024**: {len(cicdiad2024_columns)} columns (descriptive naming)
    - **Common Features Identified**: {len(feature_mapping_data)} mappable features

    **Feature Mapping Strategy:**

    Based on manual analysis, I've identified {len(feature_mapping_data)} common 
    features that can be mapped between the datasets:

    - **Direct mappings** ({len([f for f in feature_mapping_data if f[3] == "direct"])} features): Identical concepts with different names
    - **Composite mappings** ({len([f for f in feature_mapping_data if f[3] == "composite_sum"])} features): Features that combine multiple columns 
    - **Engineered mappings** ({len([f for f in feature_mapping_data if f[3] == "engineered"])} features): Features requiring calculation

    **Key Mapping Categories:**

    1. **Flow Timing Features**: Duration, packet rates, inter-arrival times
    2. **Packet Size Statistics**: Min, max, mean, std deviation of packet lengths
    3. **TCP Flag Counts**: All major TCP flags (FIN, SYN, RST, PSH, ACK, ECE, CWR, URG)
    4. **Flow Aggregates**: Total packets, bytes, and derived rates
    5. **Header Information**: Combined header lengths across directions
    6. **Labels**: Target classification variables

    I have performed this mapping this way to enable cross-dataset training and robust model evaluation.
    """

    mo.md(analysis_md)
    return (feature_mapping_data,)


@app.cell
def _(feature_mapping_data):
    # Create detailed feature mapping table
    import polars as pl

    # Convert the mapping data to a DataFrame for better display
    mapping_df = pl.DataFrame(
        {
            "Common Feature": [f[0] for f in feature_mapping_data],
            "CICIOT2023 Features": [f[1] for f in feature_mapping_data],
            "CICDIAD2024 Features": [f[2] for f in feature_mapping_data],
            "Mapping Type": [f[3] for f in feature_mapping_data],
            "Rationale": [f[4] for f in feature_mapping_data],
        }
    )

    # Display the mapping table
    mapping_df
    return


@app.cell
def _(mo):
    mo.md(r"""#### Exploratory Data Analysis""")
    return


@app.cell
def _(ciciot2023_combined_path, get_dataset_preview):
    # Get CICIOT 2023 dataset preview
    ciciot2023_preview = get_dataset_preview(ciciot2023_combined_path, 100000)
    ciciot2023_preview
    return


@app.cell
def _(cicdiad2024_combined_path, get_dataset_preview):
    # Get CIC DIAD 2024 dataset preview
    cicdiad2024_preview = get_dataset_preview(cicdiad2024_combined_path, 100000)
    cicdiad2024_preview
    return


@app.cell
def _(mo):
    # Summary of key findings from exploratory data analysis

    eda_summary = """
    #### Key EDA Findings

    **CICIOT 2023 Dataset Characteristics:**
    - Primarily numeric features (Float64 types)
    - Minimal categorical data
    - Focus on IoT network behavior patterns
    - Features appear to be engineered/aggregated metrics

    **CIC DIAD 2024 Dataset Characteristics:**
    - Mix of numeric (Float64/Int64) and categorical (String) features
    - More detailed flow-level information
    - Includes metadata like IP addresses, timestamps, protocols
    - Raw network flow features with computed statistics

    **Data Quality Observations:**
    - Both datasets have 0% missing values (excellent completeness)
    - Large scale datasets (46M+ and 19M+ rows respectively)
    - Different feature engineering approaches require careful mapping
    - Ready for machine learning pipeline development
    """

    mo.md(eda_summary)
    return


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
