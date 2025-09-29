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
    mo.md(
        r"""
    # University of London MSc Cyber Security
    # Project Experiment Report
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Data Sourcing and Analysis""")
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
        DatasetAnalyzer,
        DatasetComparator,
        Path,
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
    return (settings,)


@app.cell
def _(mo):
    mo.md(r"""### Dataset Download and Extraction""")
    return


@app.cell
def _(settings):
    # Check for existing parquet files
    print("=== PARQUET DATASET STATUS ===")

    # Check for combined parquet files
    ciciot2023_combined_path = settings.parquet_path / "ciciot2023_combined.parquet"
    cicdiad2024_combined_path = settings.parquet_path / "cicdiad2024_combined.parquet"

    # Check for harmonized parquet files
    ciciot2023_harmonized_path = (
        settings.processed_parquet_path / "revised_ciciot2023.parquet"
    )
    cicdiad2024_harmonized_path = (
        settings.processed_parquet_path / "revised_cicdiad2024.parquet"
    )

    def print_parquet_status(name, path):
        if path.exists():
            print(f"{name}: Available ({path.name})")
        else:
            print(f"{name}: Not found ({path.name})")

    print("\nCombined Parquet Files:")
    print_parquet_status("CICIOT 2023 Combined", ciciot2023_combined_path)
    print_parquet_status("CIC DIAD 2024 Combined", cicdiad2024_combined_path)

    print("\nHarmonized Parquet Files:")
    print_parquet_status("CICIOT 2023 Harmonized", ciciot2023_harmonized_path)
    print_parquet_status("CIC DIAD 2024 Harmonized", cicdiad2024_harmonized_path)
    return (
        cicdiad2024_combined_path,
        cicdiad2024_harmonized_path,
        ciciot2023_combined_path,
        ciciot2023_harmonized_path,
    )


@app.cell
def _(mo):
    mo.md(r"""### Dataset Statistics and Analysis""")
    return


@app.cell
def _(analyze_dataset_file, ciciot2023_combined_path):
    # Analyze base CICIOT 2023 dataset
    if ciciot2023_combined_path and ciciot2023_combined_path.exists():
        ciciot2023_base_analyzer = analyze_dataset_file(
            ciciot2023_combined_path, "CICIOT 2023 Base Dataset"
        )
    else:
        ciciot2023_base_analyzer = None
        print("CICIOT 2023 base dataset not available")
    return


@app.cell
def _(analyze_dataset_file, cicdiad2024_combined_path):
    # Analyze base CIC DIAD 2024 dataset
    if cicdiad2024_combined_path and cicdiad2024_combined_path.exists():
        cicdiad2024_base_analyzer = analyze_dataset_file(
            cicdiad2024_combined_path, "CIC DIAD 2024 Base Dataset"
        )
    else:
        cicdiad2024_base_analyzer = None
        print("CIC DIAD 2024 base dataset not available")
    return


@app.cell
def _(mo):
    mo.md(r"""### Dataset Comparison""")
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
    mo.md(r"""### Complete Column Inventory""")
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
    mo.md(r"""### Complete Feature Mapping Table""")
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
    ### Feature Mapping Analysis

    **Dataset Overview:**

    - **CICIOT 2023**: {len(ciciot2023_columns)} columns (abbreviated naming)
    - **CIC DIAD 2024**: {len(cicdiad2024_columns)} columns (descriptive naming)
    - **Common Features Identified**: {len(feature_mapping_data)} mappable features

    **Feature Mapping Strategy:**

    Based on manual analysis, I have identified {len(feature_mapping_data)} common 
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
    return


@app.cell
def _():
    # Display feature mapping table using module function for cleaner code
    from src.knowledge_distillation_ensemble.ml.data.feature_mappings import (
        get_feature_mapping_df,
    )

    # Get the DataFrame from module (cleaner than manual construction)
    mapping_df = get_feature_mapping_df()

    print("Complete Feature Mapping Table:")
    print(f"Showing {len(mapping_df)} feature mappings for cross-dataset alignment")
    print("=" * 80)

    # Display the mapping table
    mapping_df
    return


@app.cell
def _(mo):
    mo.md(r"""### Exploratory Data Analysis""")
    return


@app.cell
def _(get_dataset_preview, harmonized_cic23_path):
    # Get harmonized CICIOT 2023 dataset preview
    if harmonized_cic23_path and harmonized_cic23_path.exists():
        ciciot2023_preview = get_dataset_preview(harmonized_cic23_path, 100000)
    else:
        print("CICIOT 2023 harmonized dataset not available for preview")
    return (ciciot2023_preview,)


@app.cell
def _(ciciot2023_preview):
    ciciot2023_preview
    return


@app.cell
def _(get_dataset_preview, harmonized_diad_path):
    # Get harmonized CIC DIAD 2024 dataset preview
    if harmonized_diad_path and harmonized_diad_path.exists():
        cicdiad2024_preview = get_dataset_preview(harmonized_diad_path, 100000)
    else:
        print("CIC DIAD 2024 harmonized dataset not available for preview")
    return (cicdiad2024_preview,)


@app.cell
def _(cicdiad2024_preview):
    cicdiad2024_preview
    return


@app.cell
def _(mo):
    # Summary of key findings from exploratory data analysis

    eda_summary = """
    #### Key EDA Findings I observed

    **CICIOT 2023 Dataset Characteristics:**

    - Primarily numeric features (Float64 types)
    - Minimal categorical data
    - Focus on IoT network behaviour patterns
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
    mo.md(r"""### Data Preprocessing""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    #### Dataset Harmonization Implementation

    Based on the feature mapping analysis above, dataset harmonization has been 
    completed with the following approach:

    **Key Features:**

    - **Robust column matching**: Uses normalised names and multiple candidates
    - **Null-safe operations**: Handles missing values gracefully 
    - **Binarised flag encoding**: Converts TCP flag counts to 0/1 values
    - **Composite feature engineering**: Combines forward/backward directions
    - **Memory-efficient processing**: Using Polars LazyFrame operations

    **Output**: 23 harmonized features covering flow timing, packet statistics, 
    TCP flags, volume metrics, and labels as per the mapping analysis above.
    """
    )
    return


@app.cell
def _(settings):
    # Note: Dataset harmonization completed - using existing revised parquet files
    print("Harmonized Dataset Configuration")
    print(f"Processed data directory: {settings.processed_parquet_path}")
    print("Using pre-generated revised harmonized datasets")
    return


@app.cell
def _(cicdiad2024_harmonized_path, ciciot2023_harmonized_path):
    # Use existing revised harmonized datasets (streamlined - no regeneration needed)
    from src.knowledge_distillation_ensemble.ml.data.dataset_analyzer import (
        analyze_label_distributions,
    )

    print("Dataset Harmonization Status - Using Revised Files")
    print("=" * 55)

    if ciciot2023_harmonized_path.exists() and cicdiad2024_harmonized_path.exists():
        print("Revised harmonized datasets available")
        harmonized_cic23_path = ciciot2023_harmonized_path
        harmonized_diad_path = cicdiad2024_harmonized_path

        # Quick validation using existing analyzer functions
        print("\nQuick Validation:")
        ciciot_analysis = analyze_label_distributions(harmonized_cic23_path, "label")
        cicdiad_analysis = analyze_label_distributions(harmonized_diad_path, "label")

        if "error" not in ciciot_analysis and "error" not in cicdiad_analysis:
            print(
                f"  • CICIOT2023: {ciciot_analysis['total_records']:,} records, {len(ciciot_analysis['unique_labels'])} label types"
            )
            print(
                f"  • CICDIAD2024: {cicdiad_analysis['total_records']:,} records, {len(cicdiad_analysis['unique_labels'])} label types"
            )

            # Check for Mirai preservation
            mirai_in_ciciot = "Mirai" in ciciot_analysis["unique_labels"]
            mirai_in_cicdiad = "Mirai" in cicdiad_analysis["unique_labels"]
            print(
                f"  • Mirai category preserved: {'yes' if mirai_in_ciciot and mirai_in_cicdiad else 'no'}"
            )
        else:
            print("Analysis validation encountered issues")
    else:
        print("Revised harmonized datasets not found!")
        harmonized_cic23_path = None
        harmonized_diad_path = None

    print(f"\nDataset paths:")
    print(f"  • CICIOT2023: {harmonized_cic23_path}")
    print(f"  • CICDIAD2024: {harmonized_diad_path}")
    return harmonized_cic23_path, harmonized_diad_path


@app.cell
def _(analyze_dataset_file, harmonized_cic23_path):
    # Analyze harmonized CICIOT 2023 dataset
    if harmonized_cic23_path and harmonized_cic23_path.exists():
        ciciot2023_analyzer = analyze_dataset_file(
            harmonized_cic23_path, "CICIOT 2023 Harmonized Dataset"
        )
    else:
        ciciot2023_analyzer = None
        print("CICIOT 2023 harmonized dataset not available")
    return


@app.cell
def _(analyze_dataset_file, harmonized_diad_path):
    # Analyze harmonized CIC DIAD 2024 dataset
    if harmonized_diad_path and harmonized_diad_path.exists():
        cicdiad2024_analyzer = analyze_dataset_file(
            harmonized_diad_path, "CIC DIAD 2024 Harmonized Dataset"
        )
    else:
        cicdiad2024_analyzer = None
        print("CIC DIAD 2024 harmonized dataset not available")
    return


@app.cell
def _(mo):
    mo.md(r"""#### Harmonized Dataset Previews""")
    return


@app.cell
def _(get_dataset_preview, harmonized_cic23_path, mo):
    # Show harmonized CICIOT2023 preview
    if harmonized_cic23_path and harmonized_cic23_path.exists():
        print("CICIOT2023 Harmonized Dataset Preview:")
        cic23_harmonized = get_dataset_preview(harmonized_cic23_path, 10)
    else:
        mo.md("CICIOT2023 harmonized dataset not available")
        cic23_harmonized = None
    return (cic23_harmonized,)


@app.cell
def _(cic23_harmonized):
    cic23_harmonized
    return


@app.cell
def _(get_dataset_preview, harmonized_diad_path, mo):
    # Show harmonized CICDIAD2024 preview
    if harmonized_diad_path and harmonized_diad_path.exists():
        print("CICDIAD2024 Harmonized Dataset Preview:")
        diad_harmonized = get_dataset_preview(harmonized_diad_path, 10)
    else:
        mo.md("CICDIAD2024 harmonized dataset not available")
        diad_harmonized = None
    return (diad_harmonized,)


@app.cell
def _(diad_harmonized):
    diad_harmonized
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Aggregate Visualizations

    We visualize small, meaningful summaries:
    - Class imbalance (counts and percent) per dataset
    - Mapping validation (CICIOT2023) with row-normalized heatmap and purity
    - Cross-dataset shift via feature medians/IQR and KS distance
    """
    )
    return


@app.cell
def _():
    # Lightweight helpers
    from src.knowledge_distillation_ensemble.ml.evaluation.visualizations import (
        seaborn_init,
        compute_label_stats,
        chart_label_stats,
        compute_mapping_percent,
        chart_mapping_heatmap,
        compute_feature_summary,
        chart_feature_summary,
        compute_ks_table,
        chart_ks_table,
        show_feature_summary_for_datasets,
        # Newly added utilities for encoded labels
        compute_label_binary_stats,
        chart_label_binary,
        compute_label_multiclass_stats,
        chart_label_multiclass,
    )

    seaborn_init()
    return (
        chart_ks_table,
        chart_label_binary,
        chart_label_multiclass,
        chart_label_stats,
        chart_mapping_heatmap,
        compute_ks_table,
        compute_label_binary_stats,
        compute_label_multiclass_stats,
        compute_label_stats,
        compute_mapping_percent,
        show_feature_summary_for_datasets,
    )


@app.cell
def _(
    Path,
    chart_label_stats,
    compute_label_stats,
    harmonized_cic23_path,
    harmonized_diad_path,
):
    # Class imbalance
    if harmonized_cic23_path and Path(harmonized_cic23_path).exists():
        stats23 = compute_label_stats(str(harmonized_cic23_path))
        chart_label_stats(stats23, "CICIOT2023 (harmonized)")
    else:
        print("CICIOT2023: harmonized parquet not available")

    if harmonized_diad_path and Path(harmonized_diad_path).exists():
        stats24 = compute_label_stats(str(harmonized_diad_path))
        chart_label_stats(stats24, "CICDIAD2024 (harmonized)")
    else:
        print("CICDIAD2024: harmonized parquet not available")
    return


@app.cell
def _(
    Path,
    chart_label_binary,
    compute_label_binary_stats,
    harmonized_cic23_path,
    harmonized_diad_path,
):
    # Encoded label distributions placed directly under harmonized charts
    # 1) label_binary (0=Benign, 1=Attack)
    if harmonized_cic23_path and Path(harmonized_cic23_path).exists():
        bin23 = compute_label_binary_stats(str(harmonized_cic23_path))
        chart_label_binary(bin23, "CICIOT2023 label_binary")
    else:
        print("CICIOT2023: missing for label_binary")

    if harmonized_diad_path and Path(harmonized_diad_path).exists():
        bin24 = compute_label_binary_stats(str(harmonized_diad_path))
        chart_label_binary(bin24, "CICDIAD2024 label_binary")
    else:
        print("CICDIAD2024: missing for label_binary")
    return


@app.cell
def _(
    Path,
    chart_label_multiclass,
    compute_label_multiclass_stats,
    harmonized_cic23_path,
    harmonized_diad_path,
):
    # 2) label_multiclass (0..7)
    if harmonized_cic23_path and Path(harmonized_cic23_path).exists():
        mc23 = compute_label_multiclass_stats(str(harmonized_cic23_path))
        chart_label_multiclass(mc23, "CICIOT2023 label_multiclass")
    else:
        print("CICIOT2023: missing for label_multiclass")

    if harmonized_diad_path and Path(harmonized_diad_path).exists():
        mc24 = compute_label_multiclass_stats(str(harmonized_diad_path))
        chart_label_multiclass(mc24, "CICDIAD2024 label_multiclass")
    else:
        print("CICDIAD2024: missing for label_multiclass")
    return


@app.cell
def _(
    Path,
    chart_mapping_heatmap,
    compute_mapping_percent,
    harmonized_cic23_path,
):
    # Mapping validation (CICIOT2023)
    if harmonized_cic23_path and Path(harmonized_cic23_path).exists():
        ct_top, diag = compute_mapping_percent(
            str(harmonized_cic23_path), top_original=15
        )
        purity = float(diag["pct"].mean()) if diag.height else float("nan")
        print(f"Mapping purity (mean diagonal %): {purity:.2f}%")
        chart_mapping_heatmap(ct_top, "Original → Harmonized (row-normalized %)")
    else:
        print("CICIOT2023 harmonized parquet not available for mapping heatmap.")
    return


@app.cell
def _(Path, compute_ks_table, harmonized_cic23_path, harmonized_diad_path):
    # Derive top-8 features by KS for focused medians/IQR
    if (
        harmonized_cic23_path
        and harmonized_diad_path
        and Path(harmonized_cic23_path).exists()
        and Path(harmonized_diad_path).exists()
    ):
        _ks_df = compute_ks_table(
            str(harmonized_cic23_path),
            str(harmonized_diad_path),
            [
                "flow_bytes_per_second",
                "flow_packets_per_second",
                "packet_length_mean",
                "flow_duration",
                "average_packet_size",
                "total_packets",
                "total_bytes",
                "header_length_total",
                "fin_flag_count",
                "syn_flag_count",
                "rst_flag_count",
                "psh_flag_count",
                "ack_flag_count",
                "ece_flag_count",
                "cwr_flag_count",
                "urg_flag_count",
                "packet_length_min",
                "packet_length_max",
                "packet_length_std",
                "packet_length_range",
                "forward_packets_per_second",
                "backward_packets_per_second",
                "flow_iat_mean",
            ],
        )
        top_features = _ks_df.sort("ks", descending=True).head(8)["feature"].to_list()
    else:
        top_features = [
            "flow_bytes_per_second",
            "flow_packets_per_second",
            "packet_length_mean",
        ]
    top_features
    return (top_features,)


@app.cell
def _(
    Path,
    harmonized_cic23_path,
    harmonized_diad_path,
    show_feature_summary_for_datasets,
    top_features,
):
    # Focused medians/IQR on top-8 KS features
    show_feature_summary_for_datasets(
        str(harmonized_cic23_path)
        if harmonized_cic23_path and Path(harmonized_cic23_path).exists()
        else None,
        str(harmonized_diad_path)
        if harmonized_diad_path and Path(harmonized_diad_path).exists()
        else None,
        top_features,
        title="Feature medians (points) and IQR (bars) — top-8 features by KS",
    )
    return


@app.cell
def _(
    Path,
    chart_ks_table,
    compute_ks_table,
    harmonized_cic23_path,
    harmonized_diad_path,
):
    # KS distance charts (two visuals) for all 23 features
    all_features = [
        "flow_bytes_per_second",
        "flow_packets_per_second",
        "packet_length_mean",
        "flow_duration",
        "average_packet_size",
        "total_packets",
        "total_bytes",
        "header_length_total",
        "fin_flag_count",
        "syn_flag_count",
        "rst_flag_count",
        "psh_flag_count",
        "ack_flag_count",
        "ece_flag_count",
        "cwr_flag_count",
        "urg_flag_count",
        "packet_length_min",
        "packet_length_max",
        "packet_length_std",
        "packet_length_range",
        "forward_packets_per_second",
        "backward_packets_per_second",
        "flow_iat_mean",
    ]
    if (
        harmonized_cic23_path
        and harmonized_diad_path
        and Path(harmonized_cic23_path).exists()
        and Path(harmonized_diad_path).exists()
    ):
        ks_df = compute_ks_table(
            str(harmonized_cic23_path), str(harmonized_diad_path), all_features
        )
        chart_ks_table(ks_df, "KS distance: CICIOT2023 vs CICDIAD2024")
    else:
        print("Need both harmonized datasets for KS comparison.")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Data Processing Summary and Key Insights of the Harmonized Datasets

    This data processing section synthesizes the outcomes of data harmonization, summary statistics, and the visual analyses efforts to set up the machine learning plan for the teacher and student model knowledge distillation ensemble approach.

    #### 1) Harmonization and data integrity
    - Feature schema: I have implemented 23 standardized flow features in both datasets to ensure consistency when testing the performance of the machine learning analysis approach to the new dataset CIC DIAD2024 to test for robustness.
    - Labels:
        - `label` contains harmonized semantic classes; `original_label` preserves raw labels for traceability.
    - Outcome: With the preparation process, I have achieved a consistent cross-dataset schema with lineage to originals — suitable for cross-domain ML and error analysis.

    #### 2) Class imbalance (counts and percent)
    - I have observed that CICIOT2023 and CICDIAD2024 are both heavily imbalanced as DoS/DDoS dominate.
    - Minority classes (e.g., Brute_Force, Spoofing, Reconnaissance) represent a tiny fraction of samples.
    - Implications:
      - I will be using stratified splits and macro-averaged metrics (macro-F1, macro-PR AUC).
      - I will apply class weights or rebalancing strategies; consider calibrated decision thresholds.
      - I will prefer cost-sensitive training; monitor minority recall explicitly.

    #### 3) Original → Harmonized mapping (CICIOT2023 and CICDIAD2024)
    - The row-normalized heatmap shows diagonal dominance where I have mapped original attack types cleanly to harmonized classes.
    - I have also implemented a semantically coherent reclassification for labels across both datasets (e.g., DNS_Spoofing/MITM-ArpSpoofing → Spoofing; VulnerabilityScan → Reconnaissance).
    - Implications:
      - Minimal label leakage/ambiguity across datasets.
      - Downstream evaluation will reflect true performance of machine learning security approach across class semantics rather than dataset-specific taxonomy.

    #### 4) Cross-dataset shift — KS distance (CICIOT2023 vs CICDIAD2024)
    - KS distance highlights which features change most across datasets (unitless, 0–1 scale):
      - Highest shift features include packet/flow timing and TCP-flag-related counts.
      - Lower shift features include flow-bytes-per-second and some size aggregates.
    - Implications:
      - I will apply robust scaling per feature (e.g., quantile/robust scaler) before model training.

    #### 5) Feature medians and IQR (top-8 by KS)
    - Per-dataset medians and dispersion reveal which distributions moved and by how much:
      - Flow timing (e.g., `flow_duration`, `flow_iat_mean`) shows large median and IQR changes.
      - Several TCP flag counts shift in location but retain compact dispersion in one dataset.
    - Implications:
      - This means I need to be intentional about my normalization choice; and will consider log/robust transforms where heavy-tailed.

    #### 6) Next steps for the Machine Learning Modelling
    - Data pipeline:
      - I will perform robust scaling (per-feature) + optional log transforms for skewed features.
      - I will perform stratified cross-validation (CV); and focus on using macro-F1 and PR-AUC as primary evaluation metrics.
      - I will train on CICIOT2023, validate on held-out CICIOT2023, and test on CICDIAD2024 to measure robustness.
    - Modeling:
      - I will begin with setting up the thorougly calibrated and knowledgeable teacher model
      - I will proceed with setting up the ensemble student model
      - I will implement the knowledge distillation process to transfer knowledge from the teacher to the student model.
      - I will evaluate the teacher model, the trained student model, a non-trained student model, and a baseline model on the CICDIAD2024 to assess performance and robustness.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Machine Learning Pipeline""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Machine-Learning-Ready Datasets""")
    return


@app.cell
def _():
    # Create (or reuse) analysis_* parquets with capped per-class samples
    # - 200k for label_multiclass == 0 (Benign)
    # - up to 100k for each of 1..7
    from src.knowledge_distillation_ensemble.ml.data.analysis_builder import (
        create_analysis_parquet,
    )

    analysis_cic23_path = create_analysis_parquet("ciciot2023", seed=42, overwrite=True)
    analysis_diad_path = create_analysis_parquet("cicdiad2024", seed=42, overwrite=True)

    print("Analysis datasets ready:")
    print(f"  • CICIOT2023: {analysis_cic23_path}")
    print(f"  • CICDIAD2024: {analysis_diad_path}")
    return analysis_cic23_path, analysis_diad_path


@app.cell
def _(
    Path,
    analysis_cic23_path,
    analysis_diad_path,
    chart_label_binary,
    chart_label_multiclass,
    chart_label_stats,
    compute_label_binary_stats,
    compute_label_multiclass_stats,
    compute_label_stats,
):
    # Label distributions on analysis_* datasets
    def _maybe_chart(path: Path, title_prefix: str):
        if path and Path(path).exists():
            # Text labels
            stats = compute_label_stats(str(path))
            chart_label_stats(stats, f"{title_prefix} (analysis)")
            # Binary
            bstats = compute_label_binary_stats(str(path))
            chart_label_binary(bstats, f"{title_prefix} label_binary")
            # Multiclass
            mstats = compute_label_multiclass_stats(str(path))
            chart_label_multiclass(mstats, f"{title_prefix} label_multiclass")
        else:
            print(f"Missing analysis dataset: {path}")

    _maybe_chart(analysis_cic23_path, "CICIOT2023")
    _maybe_chart(analysis_diad_path, "CICDIAD2024")
    return


@app.cell
def _(mo):
    mo.md(r"""### Teacher Model Training""")
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


@app.cell
def _():
    # Utility: regenerate harmonized files if needed (optional manual run)
    # from src.knowledge_distillation_ensemble.ml.data.feature_harmonizer import (
    #     FeatureHarmonizer,
    # )

    # fh = FeatureHarmonizer()

    # cic23_in = settings.parquet_path / "ciciot2023_combined.parquet"
    # diad_in = settings.parquet_path / "cicdiad2024_combined.parquet"
    # cic23_out = settings.processed_parquet_path / "revised_ciciot2023.parquet"
    # diad_out = settings.processed_parquet_path / "revised_cicdiad2024.parquet"

    # print("Regeneration helper ready. Call fh.harmonize_dataset(...) as needed.")
    # print(f"CICIOT2023 in/out: {cic23_in} -> {cic23_out}")
    # print(f"CICDIAD2024 in/out: {diad_in} -> {diad_out}")
    # return fh, cic23_in, diad_in, cic23_out, diad_out

    print("Helper code for me to comment when I need it")
    return


if __name__ == "__main__":
    app.run()
