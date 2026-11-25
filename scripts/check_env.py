"""Stage 0 end-to-end environment verification script for LakePulse-AI."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def check_python_and_config() -> None:
    """Verify Python 3.11.x and dual-mode configuration resolution."""
    from lakepulse.config import get_settings

    ver = sys.version_info
    if (ver.major, ver.minor) != (3, 11):
        raise RuntimeError(
            f"Expected Python 3.11.x, got {ver.major}.{ver.minor}.{ver.micro}"
        )

    local_cfg = get_settings(mode="local")
    cluster_cfg = get_settings(mode="cluster")
    local_uri = local_cfg.resolve_table_uri("dwd", "dwd_daily_item_sales")
    cluster_uri = cluster_cfg.resolve_table_uri("dwd", "dwd_daily_item_sales")

    assert "lakehouse" in local_uri, f"Unexpected local URI: {local_uri}"
    assert cluster_uri == "hdfs://localhost:9000/lakepulse/dwd/dwd_daily_item_sales", (
        f"Unexpected cluster URI: {cluster_uri}"
    )
    print(f"[PASS] Python {ver.major}.{ver.minor}.{ver.micro} & dual-mode config OK")


def check_spark_and_duckdb() -> None:
    """Verify PySpark 3.5, JDK 17 binding, @pandas_udf, Parquet I/O, and DuckDB interop."""
    import duckdb
    import pyspark
    from pyspark.sql import Window
    from pyspark.sql import functions as F
    from pyspark.sql.functions import pandas_udf

    from lakepulse.spark.io import read_parquet, write_parquet
    from lakepulse.spark.session import build_spark_session

    spark = build_spark_session("LakePulse-EnvCheck")
    try:
        @pandas_udf("double")
        def buffered_sales_udf(sales: pd.Series) -> pd.Series:
            return sales.astype(float) * 1.1

        rows = [
            ("CA_1", "FOODS_1_001", 1, 10.0),
            ("CA_1", "FOODS_1_001", 2, 14.0),
            ("CA_1", "FOODS_1_001", 3, 12.0),
            ("CA_2", "FOODS_1_002", 1, 6.0),
            ("CA_2", "FOODS_1_002", 2, 8.0),
        ]
        df = spark.createDataFrame(rows, ["store_id", "item_id", "day_idx", "sales"])

        win = (
            Window.partitionBy("store_id", "item_id")
            .orderBy("day_idx")
            .rowsBetween(-1, 0)
        )
        df_feat = df.withColumn("rolling_avg_2d", F.avg("sales").over(win)).withColumn(
            "buffered_sales", buffered_sales_udf(F.col("sales"))
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            parquet_path = Path(tmp_dir) / "sample_sales"
            write_parquet(df_feat, parquet_path, partition_cols=["store_id"], mode="overwrite")

            spark_roundtrip = read_parquet(spark, parquet_path)
            spark_count = spark_roundtrip.count()
            assert spark_count == 5, f"Expected 5 rows from Spark read, got {spark_count}"

            glob_pattern = str(parquet_path / "**" / "*.parquet")
            con = duckdb.connect(":memory:")
            try:
                duck_res = con.execute(
                    f"""
                    SELECT
                        COUNT(*) AS cnt,
                        ROUND(AVG(rolling_avg_2d), 4) AS avg_val,
                        ROUND(SUM(buffered_sales), 2) AS sum_buf
                    FROM read_parquet('{glob_pattern}', hive_partitioning=true)
                    """
                ).fetchone()
            finally:
                con.close()

            assert duck_res is not None and duck_res[0] == 5, (
                f"Expected 5 rows from DuckDB, got {duck_res}"
            )
            assert abs(float(duck_res[2]) - 55.0) < 1e-6, (
                f"Unexpected @pandas_udf sum: {duck_res[2]}"
            )
            java_home = os.environ.get("JAVA_HOME", "")
            print(
                f"[PASS] PySpark {pyspark.__version__} (JAVA_HOME={java_home}) "
                f"+ @pandas_udf + Parquet + DuckDB {duckdb.__version__} OK "
                f"(rows={duck_res[0]}, avg_ma={duck_res[1]})"
            )
    finally:
        spark.stop()


def check_dbt() -> None:
    """Verify dbt-core and dbt-duckdb adapter availability."""
    import dbt.adapters.duckdb as dbt_duckdb
    import dbt.version as dbt_version

    assert dbt_duckdb is not None
    print(f"[PASS] dbt-core {dbt_version.__version__} + dbt-duckdb OK")


def check_tensorflow_wide_and_deep() -> None:
    """Verify TensorFlow 2.x can build and execute a Wide & Deep regression model."""
    import numpy as np
    import tensorflow as tf

    wide_in = tf.keras.Input(shape=(4,), name="wide_features")
    deep_in = tf.keras.Input(shape=(6,), name="deep_features")

    wide_out = tf.keras.layers.Dense(1, use_bias=False)(wide_in)
    deep_h = tf.keras.layers.Dense(16, activation="relu")(deep_in)
    deep_h = tf.keras.layers.Dense(8, activation="relu")(deep_h)
    deep_out = tf.keras.layers.Dense(1)(deep_h)

    merged = tf.keras.layers.Add()([wide_out, deep_out])
    output = tf.keras.layers.ReLU()(merged)

    model = tf.keras.Model(inputs=[wide_in, deep_in], outputs=output)
    preds = model([np.ones((3, 4), dtype=np.float32), np.ones((3, 6), dtype=np.float32)],
                  training=False).numpy()

    assert preds.shape == (3, 1), f"Unexpected prediction shape: {preds.shape}"
    assert np.isfinite(preds).all(), "TensorFlow output contains non-finite values"
    print(f"[PASS] TensorFlow {tf.__version__} Wide & Deep forward pass OK")


def check_ml_and_gateway_deps() -> None:
    """Verify LightGBM, FastAPI, and sqlglot AST safety parsing."""
    import fastapi
    import lightgbm as lgb
    import sqlglot
    from sqlglot import exp

    select_ast = sqlglot.parse_one("SELECT store_id, SUM(sales) FROM ads_replenishment_plan")
    drop_ast = sqlglot.parse_one("DROP TABLE ads_replenishment_plan")

    assert isinstance(select_ast, exp.Select)
    assert not isinstance(drop_ast, exp.Select)
    print(
        f"[PASS] LightGBM {lgb.__version__}, FastAPI {fastapi.__version__}, "
        f"sqlglot {sqlglot.__version__} OK"
    )


def main() -> int:
    """Run all Stage 0 environment checks."""
    print("=== LakePulse-AI Stage 0 Environment Check ===")
    check_python_and_config()
    check_spark_and_duckdb()
    check_dbt()
    check_tensorflow_wide_and_deep()
    check_ml_and_gateway_deps()
    print("=== All Stage 0 checks passed! ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
