"""Distributed PySpark ODS ETL: M5 wide-to-long unpivot, deduplication, and Parquet landing."""

from __future__ import annotations

import re
import time
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from lakepulse.config import AppSettings, get_settings
from lakepulse.etl.audit import (
    AuditRecord,
    append_audit_records_spark,
    generate_run_id,
    utc_now_iso,
)
from lakepulse.spark.io import write_parquet
from lakepulse.spark.session import build_spark_session


def _locate_sales_csv(raw_dir: Path) -> Path:
    """Locate M5 evaluation or validation wide sales CSV."""
    eval_path = raw_dir / "sales_train_evaluation.csv"
    if eval_path.is_file():
        return eval_path
    val_path = raw_dir / "sales_train_validation.csv"
    if val_path.is_file():
        return val_path
    raise FileNotFoundError(
        f"Neither sales_train_evaluation.csv nor sales_train_validation.csv found in {raw_dir}"
    )


def _compute_null_rate(df: DataFrame, cols: list[str], total_rows: int) -> float:
    """Compute average null fraction across key columns."""
    if total_rows <= 0 or not cols:
        return 0.0
    exprs = [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in cols]
    row = df.agg(*exprs).collect()[0]
    total_nulls = sum(int(row[c] or 0) for c in cols)
    return round(total_nulls / (total_rows * len(cols)), 6)


def _process_calendar(
    spark: SparkSession,
    raw_dir: Path,
    cfg: AppSettings,
    run_id: str,
) -> AuditRecord:
    """Clean and land ods_calendar Parquet table."""
    started_at = utc_now_iso()
    t0 = time.perf_counter()

    cal_path = raw_dir / "calendar.csv"
    df_raw = spark.read.option("header", "true").option("inferSchema", "true").csv(str(cal_path))
    input_rows = int(df_raw.count())

    df_clean = df_raw.select(
        F.col("date").cast("string").alias("date"),
        F.col("wm_yr_wk").cast("int").alias("wm_yr_wk"),
        F.col("weekday").cast("string").alias("weekday"),
        F.col("wday").cast("int").alias("wday"),
        F.col("month").cast("int").alias("month"),
        F.col("year").cast("int").alias("year"),
        F.col("d").cast("string").alias("d"),
        *[
            F.when(F.trim(F.col(c).cast("string")) == "", F.lit(None).cast("string"))
            .otherwise(F.col(c).cast("string"))
            .alias(c)
            for c in ("event_name_1", "event_type_1", "event_name_2", "event_type_2")
        ],
        F.col("snap_CA").cast("int").alias("snap_CA"),
        F.col("snap_TX").cast("int").alias("snap_TX"),
        F.col("snap_WI").cast("int").alias("snap_WI"),
    ).dropDuplicates(["d"])

    output_rows = int(df_clean.count())
    null_rate = _compute_null_rate(df_clean, ["date", "wm_yr_wk", "d"], output_rows)
    write_parquet(df_clean, cfg.resolve_table_uri("ods", "ods_calendar"), mode="overwrite")

    duration_sec = round(time.perf_counter() - t0, 3)
    finished_at = utc_now_iso()
    dedup_cnt = max(0, input_rows - output_rows)
    status = "WARN" if dedup_cnt > 0 else "SUCCESS"
    msg = f"ods_calendar landed cleanly (deduplicated={dedup_cnt})."

    return AuditRecord(
        run_id=run_id,
        pipeline_stage="stage1_ods_etl",
        table_name="ods_calendar",
        started_at=started_at,
        finished_at=finished_at,
        duration_sec=duration_sec,
        input_rows=input_rows,
        output_rows=output_rows,
        null_rate=null_rate,
        status=status,
        message=msg,
    )


def _process_sell_prices(
    spark: SparkSession,
    raw_dir: Path,
    cfg: AppSettings,
    run_id: str,
    *,
    full_dataset: bool,
    sample_stores: list[str],
) -> AuditRecord:
    """Filter, deduplicate, and land ods_sell_prices Parquet table."""
    started_at = utc_now_iso()
    t0 = time.perf_counter()

    prices_path = raw_dir / "sell_prices.csv"
    df_raw = (
        spark.read.option("header", "true").option("inferSchema", "true").csv(str(prices_path))
    )
    if not full_dataset and sample_stores:
        df_raw = df_raw.filter(F.col("store_id").isin(sample_stores))

    df_typed = df_raw.select(
        F.col("store_id").cast("string").alias("store_id"),
        F.col("item_id").cast("string").alias("item_id"),
        F.col("wm_yr_wk").cast("int").alias("wm_yr_wk"),
        F.col("sell_price").cast("double").alias("sell_price"),
    )
    input_rows = int(df_typed.count())

    df_dedup = df_typed.groupBy("store_id", "item_id", "wm_yr_wk").agg(
        F.round(F.avg("sell_price"), 4).alias("sell_price")
    )
    output_rows = int(df_dedup.count())
    null_rate = _compute_null_rate(
        df_dedup, ["store_id", "item_id", "wm_yr_wk", "sell_price"], output_rows
    )
    write_parquet(df_dedup, cfg.resolve_table_uri("ods", "ods_sell_prices"), mode="overwrite")

    duration_sec = round(time.perf_counter() - t0, 3)
    finished_at = utc_now_iso()
    dedup_removed = max(0, input_rows - output_rows)
    status = "WARN" if dedup_removed > 0 else "SUCCESS"
    msg = (
        f"ods_sell_prices removed {dedup_removed} duplicate (store_id, item_id, wm_yr_wk) rows."
        if dedup_removed > 0
        else "ods_sell_prices landed cleanly."
    )

    return AuditRecord(
        run_id=run_id,
        pipeline_stage="stage1_ods_etl",
        table_name="ods_sell_prices",
        started_at=started_at,
        finished_at=finished_at,
        duration_sec=duration_sec,
        input_rows=input_rows,
        output_rows=output_rows,
        null_rate=null_rate,
        status=status,
        message=msg,
    )


def _process_sales_long(
    spark: SparkSession,
    raw_dir: Path,
    cfg: AppSettings,
    run_id: str,
    *,
    full_dataset: bool,
    sample_states: list[str],
    sample_stores: list[str],
) -> AuditRecord:
    """Unpivot M5 wide sales CSV to long format and land partitioned Parquet."""
    started_at = utc_now_iso()
    t0 = time.perf_counter()

    sales_path = _locate_sales_csv(raw_dir)
    df_wide = (
        spark.read.option("header", "true").option("inferSchema", "true").csv(str(sales_path))
    )
    if not full_dataset:
        if sample_states:
            df_wide = df_wide.filter(F.col("state_id").isin(sample_states))
        if sample_stores:
            df_wide = df_wide.filter(F.col("store_id").isin(sample_stores))

    day_cols = sorted(
        [c for c in df_wide.columns if re.fullmatch(r"d_\d+", c)],
        key=lambda c: int(c.split("_")[1]),
    )
    if not day_cols:
        raise ValueError(f"No day columns matching d_<int> found in {sales_path}")

    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    df_cast = df_wide.select(
        *[F.col(c).cast("string").alias(c) for c in id_cols],
        *[F.col(c).cast("int").alias(c) for c in day_cols],
    )
    wide_rows = int(df_cast.count())

    df_long = df_cast.unpivot(
        ids=id_cols,
        values=day_cols,
        variableColumnName="d",
        valueColumnName="sales",
    )

    stats = df_long.agg(
        F.count("*").alias("total_rows"),
        F.sum(F.when(F.col("sales") < 0, 1).otherwise(0)).alias("neg_rows"),
        F.sum(F.when(F.col("sales").isNull(), 1).otherwise(0)).alias("null_sales"),
    ).collect()[0]

    output_rows = int(stats["total_rows"] or 0)
    neg_rows = int(stats["neg_rows"] or 0)
    null_sales = int(stats["null_sales"] or 0)
    null_rate = round(null_sales / output_rows, 6) if output_rows > 0 else 0.0

    write_parquet(
        df_long,
        cfg.resolve_table_uri("ods", "ods_sales_long"),
        partition_cols=["state_id", "store_id"],
        mode="overwrite",
    )

    duration_sec = round(time.perf_counter() - t0, 3)
    finished_at = utc_now_iso()
    status = "WARN" if neg_rows > 0 else "SUCCESS"
    msg = (
        f"Unpivoted {wide_rows} wide rows x {len(day_cols)} days; "
        f"detected {neg_rows} negative sales rows deferred to DWD repair."
        if neg_rows > 0
        else f"Unpivoted {wide_rows} wide rows x {len(day_cols)} days cleanly."
    )

    return AuditRecord(
        run_id=run_id,
        pipeline_stage="stage1_ods_etl",
        table_name="ods_sales_long",
        started_at=started_at,
        finished_at=finished_at,
        duration_sec=duration_sec,
        input_rows=wide_rows,
        output_rows=output_rows,
        null_rate=null_rate,
        status=status,
        message=msg,
    )


def run_spark_ods_etl(
    settings: AppSettings | None = None,
    run_id: str | None = None,
) -> list[AuditRecord]:
    """Execute PySpark ODS ingestion and write audit metrics to ops_pipeline_audit."""
    cfg = settings or get_settings()
    batch_run_id = run_id or generate_run_id("ods")
    raw_dir = cfg.data_dir / "raw"

    data_cfg = cfg.yaml_config.get("data", {})
    full_dataset = bool(data_cfg.get("full_dataset", False))
    sample_states = [str(s) for s in data_cfg.get("sample_states", [])]
    sample_stores = [str(s) for s in data_cfg.get("sample_stores", [])]

    spark = build_spark_session("LakePulse-Stage1-ODS-ETL", cfg)
    try:
        records = [
            _process_calendar(spark, raw_dir, cfg, batch_run_id),
            _process_sell_prices(
                spark,
                raw_dir,
                cfg,
                batch_run_id,
                full_dataset=full_dataset,
                sample_stores=sample_stores,
            ),
            _process_sales_long(
                spark,
                raw_dir,
                cfg,
                batch_run_id,
                full_dataset=full_dataset,
                sample_states=sample_states,
                sample_stores=sample_stores,
            ),
        ]
        append_audit_records_spark(spark, cfg, records)
        return records
    finally:
        spark.stop()
