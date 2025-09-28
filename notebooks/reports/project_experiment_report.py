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
def _():
    base_url = (
        "https://www.kaggle.com/api/v1/datasets/download/madhavmalhotra/unb-cic-iot-dataset"
    )

    dummy_url = "versions/1/wataiData/csv/CICIoT2023/part-00000-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv"

    full_url = f"{base_url}/{dummy_url}"
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    import duckdb

    csv_glob = "/path/to/folder/*.csv"          # or abfs://container/prefix/*.csv

    with duckdb.connect() as con:
        con.execute("INSTALL httpfs; LOAD httpfs;")   # needed for abfs/s3/https
        con.execute("PRAGMA enable_object_cache=true;")

        # Create a view over all CSVs (no big materialisation yet)
        con.execute(f"CREATE OR REPLACE VIEW all_csv AS SELECT * FROM read_csv_auto('{csv_glob}', sample_size=-1);")

        # Optional: sanity check
        print(con.execute("SELECT COUNT(*) FROM all_csv").fetchone())

        # Best practice: write once to Parquet for Polars downstream
        con.execute(\"""
          COPY (SELECT * FROM all_csv)
          TO 'all.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);
        \""")
    """
    )
    return


@app.cell
def _():
    from src.knowledge_distillation_ensemble.ml.data import extract_data
    return


app._unparsable_cell(
    r"""
    /Users/chukwudumchukwuedo/.cache/kagglehub/datasets/madhavmalhotra/unb-cic-iot-dataset/versions/1 /Users/chukwudumchukwuedo/.cache/kagglehub/datasets/aymeneinformatique/cic-diad-new-2024/versions/1
    100%|
    """,
    name="_"
)


@app.cell
def _():
    # extract_data.ciciot2023
    return


@app.cell
def _():
    # extract_data.path
    return


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
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
