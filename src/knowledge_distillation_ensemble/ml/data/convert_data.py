"""
Data conversion module for Knowledge Distillation Ensemble project.

This module handles converting raw CSV data to optimized Parquet format
for efficient processing and analysis.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import duckdb
import polars as pl

from src.knowledge_distillation_ensemble.config.settings import settings

# Set up logging
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


class DataConverter:
    """Handles data format conversion and optimization."""

    def __init__(self, memory_limit: str = "15GB", threads: Optional[int] = None):
        """
        Initialize the data converter.

        Args:
            memory_limit: Memory limit for DuckDB operations
            threads: Number of threads to use (None for auto-detection)
        """
        self.memory_limit = memory_limit
        self.threads = threads
        self.output_dir = settings.data_path / "parquet"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _setup_duckdb_connection(self, con: duckdb.DuckDBPyConnection) -> None:
        """Set up DuckDB connection with optimal configuration."""
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("PRAGMA enable_object_cache=true;")
        con.execute(f"PRAGMA memory_limit='{self.memory_limit}';")

        # Additional optimizations for large datasets
        con.execute("PRAGMA temp_directory='/tmp';")  # Use tmp for spill
        con.execute("PRAGMA checkpoint_threshold='1GB';")  # Checkpoints

        if self.threads:
            con.execute(f"PRAGMA threads={self.threads};")

        logger.info(f"DuckDB configured with memory limit: {self.memory_limit}")

    def convert_csv_to_parquet(
        self,
        csv_paths: Union[str, Path, List[Union[str, Path]]],
        output_name: str,
        filter_query: Optional[str] = None,
        partition_by: Optional[List[str]] = None,
        sample_rows: Optional[int] = None,
    ) -> Path:
        """
        Convert CSV files to Parquet format with optional filtering.

        Args:
            csv_paths: Single CSV path, glob pattern, or list of CSV paths
            output_name: Name for the output parquet file/directory
            filter_query: Optional SQL WHERE clause for filtering
            partition_by: Optional list of columns to partition by
            sample_rows: Optional number of rows to sample for testing

        Returns:
            Path to the created Parquet file/directory
        """
        output_path = self.output_dir / f"{output_name}.parquet"

        # Check if Parquet file already exists
        if output_path.exists():
            logger.info(f"Parquet file already exists: {output_path}")
            logger.info("Skipping conversion. Delete the file to reconvert.")
            return output_path

        # Handle different input types
        if isinstance(csv_paths, (str, Path)):
            csv_pattern = str(csv_paths)
        elif isinstance(csv_paths, list):
            # Convert list of paths to glob pattern or multiple files
            csv_pattern = str(csv_paths[0]) if len(csv_paths) == 1 else csv_paths
        else:
            raise ValueError("csv_paths must be string, Path, or list of paths")

        logger.info(f"Converting CSV to Parquet: {csv_pattern} -> {output_path}")

        try:
            with duckdb.connect() as con:
                self._setup_duckdb_connection(con)

                # Build the SQL query
                where_clause = ""
                if filter_query:
                    where_clause = f"WHERE {filter_query}"

                limit_clause = ""
                if sample_rows:
                    limit_clause = f"LIMIT {sample_rows}"

                if isinstance(csv_pattern, list):
                    # Handle multiple files
                    union_queries = []
                    for csv_file in csv_pattern:
                        union_queries.append(
                            f"SELECT * FROM "
                            f"read_csv_auto('{csv_file}', sample_size=-1) "
                            f"{where_clause}"
                        )
                    main_query = " UNION ALL ".join(union_queries)
                    if limit_clause:
                        main_query = f"({main_query}) {limit_clause}"
                else:
                    # Handle single file or glob pattern
                    main_query = f"""
                    SELECT *
                    FROM read_csv_auto('{csv_pattern}', sample_size=-1)
                    {where_clause}
                    {limit_clause}
                    """  # Build COPY statement
                copy_options = ["FORMAT PARQUET", "COMPRESSION ZSTD"]

                if partition_by:
                    partition_cols = ", ".join(partition_by)
                    copy_options.append(f"PARTITION_BY ({partition_cols})")

                copy_statement = f"""
                COPY ({main_query})
                TO '{output_path}' ({", ".join(copy_options)});
                """

                logger.debug(f"Executing SQL: {copy_statement}")
                con.execute(copy_statement)

                logger.info(f"Successfully converted to: {output_path}")
                return output_path

        except Exception as e:
            logger.error(f"Error converting CSV to Parquet: {e}")
            raise

    def sample_csv_data(
        self, csv_path: Union[str, Path], sample_size: int = 1000
    ) -> pl.DataFrame:
        """
        Sample data from CSV for exploration and testing.

        Args:
            csv_path: Path to CSV file
            sample_size: Number of rows to sample

        Returns:
            Polars DataFrame with sampled data
        """
        logger.info(f"Sampling {sample_size} rows from {csv_path}")

        try:
            with duckdb.connect() as con:
                self._setup_duckdb_connection(con)

                query = f"""
                SELECT *
                FROM read_csv_auto('{csv_path}', sample_size=-1)
                LIMIT {sample_size}
                """

                result = con.execute(query).pl()
                logger.info(
                    f"Sampled {len(result)} rows, {len(result.columns)} columns"
                )
                return result

        except Exception as e:
            logger.error(f"Error sampling CSV data: {e}")
            raise

    def get_csv_schema(self, csv_path: Union[str, Path]) -> pl.DataFrame:
        """
        Get schema information for a CSV file.

        Args:
            csv_path: Path to CSV file

        Returns:
            DataFrame with column names and types
        """
        logger.info(f"Getting schema for {csv_path}")

        try:
            with duckdb.connect() as con:
                self._setup_duckdb_connection(con)

                query = f"""
                DESCRIBE SELECT *
                FROM read_csv_auto('{csv_path}', sample_size=1000)
                """

                schema = con.execute(query).pl()
                logger.info(f"Schema contains {len(schema)} columns")
                return schema

        except Exception as e:
            logger.error(f"Error getting CSV schema: {e}")
            raise


# Global instance for easy access
converter = DataConverter()
