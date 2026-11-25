"""Cross-platform Parquet I/O handling Windows environments without winutils.exe."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from py4j.protocol import Py4JJavaError
from pyspark.sql import DataFrame, SparkSession

REMOTE_PREFIXES = ("hdfs://", "s3://", "s3a://", "gs://", "abfs://", "abfss://")


def _is_remote_uri(path_str: str) -> bool:
    """Return True if path targets a distributed or cloud filesystem URI."""
    return path_str.lower().startswith(REMOTE_PREFIXES)


def _has_windows_winutils() -> bool:
    """Check whether winutils.exe exists under HADOOP_HOME on Windows."""
    if os.name != "nt":
        return True
    hadoop_home = os.environ.get("HADOOP_HOME", "").strip()
    if not hadoop_home:
        return False
    return (Path(hadoop_home) / "bin" / "winutils.exe").is_file()


def _use_native_hadoop_io(path_str: str) -> bool:
    """Determine if Spark's native Hadoop Parquet writer/reader can run safely."""
    return _is_remote_uri(path_str) or os.name != "nt" or _has_windows_winutils()


def _spark_df_to_arrow_table(df: DataFrame) -> pa.Table:
    """Convert a Spark DataFrame to a PyArrow Table safely, including empty tables."""
    pdf = df.toPandas()
    return pa.Table.from_pandas(pdf, preserve_index=False)


def write_parquet(
    df: DataFrame,
    path: str | Path,
    partition_cols: list[str] | None = None,
    mode: str = "overwrite",
) -> None:
    """Write a Spark DataFrame to Parquet in native distributed or Windows-safe local mode."""
    path_str = str(path)
    if _use_native_hadoop_io(path_str):
        writer = df.write.mode(mode)
        if partition_cols:
            writer = writer.partitionBy(*partition_cols)
        writer.parquet(path_str)
        return

    target = Path(path_str)
    clean_mode = mode.lower()
    if clean_mode not in {"overwrite", "append", "error", "errorifexists", "ignore"}:
        raise ValueError(f"Unsupported Parquet write mode: {mode}")

    if target.exists():
        if clean_mode in {"error", "errorifexists"}:
            raise FileExistsError(f"Target path already exists: {target}")
        if clean_mode == "ignore":
            return
        if clean_mode == "overwrite":
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()

    target.mkdir(parents=True, exist_ok=True)
    table = _spark_df_to_arrow_table(df)

    if partition_cols:
        pq.write_to_dataset(table, root_path=str(target), partition_cols=partition_cols)
    else:
        part_name = (
            "part-00000.parquet"
            if clean_mode == "overwrite"
            else f"part-{uuid.uuid4().hex}.parquet"
        )
        pq.write_table(table, target / part_name)


def read_parquet(spark: SparkSession, path: str | Path) -> DataFrame:
    """Read a Parquet dataset into a Spark DataFrame across cluster and Windows local modes."""
    path_str = str(path)
    if _use_native_hadoop_io(path_str):
        return spark.read.parquet(path_str)

    target = Path(path_str)
    if not target.exists():
        raise FileNotFoundError(f"Parquet path does not exist: {target}")

    parquet_files = (
        [str(target)]
        if target.is_file()
        else sorted(str(p) for p in target.rglob("*.parquet"))
    )
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found under: {target}")

    try:
        reader = spark.read
        if target.is_dir():
            reader = reader.option("basePath", str(target))
        return reader.parquet(*parquet_files)
    except Py4JJavaError:
        arrow_dataset = ds.dataset(str(target), format="parquet", partitioning="hive")
        pdf = arrow_dataset.to_table().to_pandas()
        return spark.createDataFrame(pdf)
