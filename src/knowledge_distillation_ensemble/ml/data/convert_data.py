import duckdb


def convert_csv_to_parquet(csv_path: str, out_dir: str):
    with duckdb.connect() as con:
        con.execute("INSTALL httpfs; LOAD httpfs;")  # + azure if using abfs://
        con.execute("PRAGMA enable_object_cache=true;")
        con.execute("PRAGMA memory_limit='1GB';")
        # All CSVs (local glob, abfs:// prefix, or list of URLs)
        con.execute(f"""
        COPY (
        SELECT col_a, col_b, ts
        FROM read_csv_auto('{csv_path}', sample_size=-1)
        WHERE col_a > 0
        )
        TO '{out_dir}' (
        FORMAT PARQUET,
        COMPRESSION ZSTD,
        PARTITION_BY (year, country)   -- optional partitions
        );
        """)


# import duckdb

# with duckdb.connect() as con:
#     con.execute("INSTALL httpfs; LOAD httpfs;")
#     con.execute("PRAGMA enable_object_cache=true;")   # speeds up repeated scans
#     # Optional tuning:
#     # con.execute("PRAGMA threads=8;")               # match your CPU
#     con.execute("PRAGMA memory_limit='1GB';")      # if you want an explicit cap

#     # Stream the filtered result straight to Parquet (no giant result in Python)
#     test_polars = duckdb.sql(f"""SELECT *
#       FROM read_csv_auto('{full_url}', sample_size=-1)
#         LIMIT 5""").pl()

# test_polars
