"""Stage 0 automated test suite verifying skeleton, dual-mode config, Spark/DuckDB, and TF."""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
import sqlglot
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf
from sqlglot import exp

from lakepulse.config import get_settings
from lakepulse.spark.io import read_parquet, write_parquet
from lakepulse.spark.session import build_spark_session

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


@pytest.fixture(scope="module")
def spark_session() -> SparkSession:
    """Provide a shared local SparkSession for Stage 0 tests."""
    spark = build_spark_session("LakePulse-Stage0-Tests")
    yield spark
    spark.stop()


def test_config_dual_mode_uri_resolution() -> None:
    """Verify local and cluster modes resolve distinct storage URIs."""
    local_cfg = get_settings(mode="local")
    cluster_cfg = get_settings(mode="cluster")

    local_uri = local_cfg.resolve_table_uri("dwd", "dwd_daily_item_sales")
    cluster_uri = cluster_cfg.resolve_table_uri("dwd", "dwd_daily_item_sales")

    assert Path(local_uri).name == "dwd_daily_item_sales"
    assert Path(local_uri).parent.name == "dwd"
    assert cluster_uri == "hdfs://localhost:9000/lakepulse/dwd/dwd_daily_item_sales"


def test_project_essential_files_exist() -> None:
    """Verify core documentation, dbt, and Docker files are present and non-empty."""
    root = get_settings().project_root
    expected_files = [
        "README.md",
        "AGENTS.md",
        "docs/architecture.md",
        "docs/rag_metrics_dictionary.md",
        "docs/archive/initial_draft.md",
        "docker/docker-compose.yml",
        "dbt_project/dbt_project.yml",
        "dbt_project/profiles.yml",
        "configs/settings.yaml",
    ]
    for rel_path in expected_files:
        file_path = root / rel_path
        assert file_path.is_file(), f"Missing required project file: {rel_path}"
        assert file_path.stat().st_size > 0, f"Project file is empty: {rel_path}"


def test_spark_parquet_and_duckdb_interop(
    spark_session: SparkSession, tmp_path: Path
) -> None:
    """Verify PySpark window + @pandas_udf -> Parquet roundtrip -> DuckDB consistency."""

    @pandas_udf("double")
    def buffered_sales_udf(sales: pd.Series) -> pd.Series:
        return sales.astype(float) * 1.1

    rows = [
        ("CA_1", "FOODS_1_001", 1, 10.0),
        ("CA_1", "FOODS_1_001", 2, 20.0),
        ("CA_2", "FOODS_1_002", 1, 5.0),
        ("CA_2", "FOODS_1_002", 2, 15.0),
    ]
    df = spark_session.createDataFrame(rows, ["store_id", "item_id", "day_idx", "sales"])
    win = Window.partitionBy("store_id").orderBy("day_idx").rowsBetween(-1, 0)
    df_feat = df.withColumn("ma_sales", F.avg("sales").over(win)).withColumn(
        "buffered_sales", buffered_sales_udf(F.col("sales"))
    )

    target_dir = tmp_path / "dws_sample"
    write_parquet(df_feat, target_dir, partition_cols=["store_id"], mode="overwrite")

    spark_back = read_parquet(spark_session, target_dir)
    spark_rows = {
        (r["store_id"], r["day_idx"]): (float(r["ma_sales"]), float(r["buffered_sales"]))
        for r in spark_back.collect()
    }
    assert spark_rows[("CA_1", 1)] == pytest.approx((10.0, 11.0))
    assert spark_rows[("CA_1", 2)] == pytest.approx((15.0, 22.0))
    assert spark_rows[("CA_2", 1)] == pytest.approx((5.0, 5.5))
    assert spark_rows[("CA_2", 2)] == pytest.approx((10.0, 16.5))

    glob_path = str(target_dir / "**" / "*.parquet")
    con = duckdb.connect(":memory:")
    try:
        duck_rows = con.execute(
            f"""
            SELECT store_id, day_idx, ma_sales, buffered_sales
            FROM read_parquet('{glob_path}', hive_partitioning=true)
            ORDER BY store_id, day_idx
            """
        ).fetchall()
    finally:
        con.close()

    expected = [
        ("CA_1", 1, 10.0, 11.0),
        ("CA_1", 2, 15.0, 22.0),
        ("CA_2", 1, 5.0, 5.5),
        ("CA_2", 2, 10.0, 16.5),
    ]
    assert len(duck_rows) == len(expected)
    for got, exp_row in zip(duck_rows, expected, strict=True):
        assert got[0] == exp_row[0]
        assert got[1] == exp_row[1]
        assert float(got[2]) == pytest.approx(exp_row[2])
        assert float(got[3]) == pytest.approx(exp_row[3])


def test_tensorflow_wide_and_deep_smoke() -> None:
    """Verify TensorFlow Wide & Deep Keras model builds and outputs finite predictions."""
    import tensorflow as tf

    wide_in = tf.keras.Input(shape=(3,), name="wide_in")
    deep_in = tf.keras.Input(shape=(5,), name="deep_in")

    wide_branch = tf.keras.layers.Dense(1)(wide_in)
    deep_branch = tf.keras.layers.Dense(8, activation="relu")(deep_in)
    deep_branch = tf.keras.layers.Dense(1)(deep_branch)
    out = tf.keras.layers.ReLU()(tf.keras.layers.Add()([wide_branch, deep_branch]))

    model = tf.keras.Model(inputs=[wide_in, deep_in], outputs=out)
    preds = model(
        [np.ones((4, 3), dtype=np.float32), np.ones((4, 5), dtype=np.float32)],
        training=False,
    ).numpy()

    assert preds.shape == (4, 1)
    assert np.isfinite(preds).all()


def test_sqlglot_select_guardrail() -> None:
    """Verify sqlglot distinguishes read-only SELECT statements from DDL/DML."""
    ok_stmt = sqlglot.parse_one("SELECT item_id FROM ads_replenishment_plan LIMIT 10")
    bad_stmt = sqlglot.parse_one("DELETE FROM ads_replenishment_plan")
    assert isinstance(ok_stmt, exp.Select)
    assert not isinstance(bad_stmt, exp.Select)
