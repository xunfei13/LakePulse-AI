"""Stage 2 orchestration pipeline for DWS diagnostics, rolling features, and DuckDB sync."""

from __future__ import annotations

import time
from pathlib import Path

import duckdb
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from lakepulse.analytics.diagnostics import (
    build_abc_profile,
    build_calendar_effect_analysis,
    build_price_elasticity_analysis,
)
from lakepulse.analytics.feature_engineering import build_rolling_features_df
from lakepulse.config import AppSettings, get_settings
from lakepulse.etl.audit import (
    AuditRecord,
    append_audit_records_spark,
    generate_run_id,
    sync_audit_table_to_duckdb,
    utc_now_iso,
)
from lakepulse.spark.io import read_parquet, write_parquet
from lakepulse.spark.session import build_spark_session

STAGE2_TABLES: tuple[tuple[str, bool], ...] = (
    ("dws_item_abc_profile", False),
    ("dws_calendar_effect_analysis", False),
    ("dws_price_elasticity_analysis", False),
    ("dws_item_rolling_features", True),
)


def _compute_null_rate(df: DataFrame, cols: list[str], total_rows: int) -> float:
    """Compute average null fraction across required columns."""
    if total_rows <= 0 or not cols:
        return 0.0
    exprs = [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in cols]
    row = df.agg(*exprs).collect()[0]
    total_nulls = sum(int(row[c] or 0) for c in cols)
    return round(total_nulls / (total_rows * len(cols)), 6)


def _persist_dws_table(
    df: DataFrame,
    cfg: AppSettings,
    run_id: str,
    table_name: str,
    input_rows: int,
    key_cols: list[str],
    *,
    partition_cols: list[str] | None = None,
    message: str = "",
) -> AuditRecord:
    """Persist a DWS Spark DataFrame to Parquet and capture audit metrics."""
    started_at = utc_now_iso()
    t0 = time.perf_counter()

    output_rows = int(df.count())
    null_rate = _compute_null_rate(df, key_cols, output_rows)
    uri = cfg.resolve_table_uri("dws", table_name)
    write_parquet(df, uri, partition_cols=partition_cols, mode="overwrite")

    duration_sec = round(time.perf_counter() - t0, 3)
    finished_at = utc_now_iso()
    status = "SUCCESS" if output_rows > 0 else "WARN"
    final_msg = message or f"{table_name} built and written cleanly."

    return AuditRecord(
        run_id=run_id,
        pipeline_stage="stage2_analytics_features",
        table_name=table_name,
        started_at=started_at,
        finished_at=finished_at,
        duration_sec=duration_sec,
        input_rows=input_rows,
        output_rows=output_rows,
        null_rate=null_rate,
        status=status,
        message=final_msg,
    )


def sync_stage2_tables_to_duckdb(settings: AppSettings | None = None) -> dict[str, int]:
    """Refresh all Stage 2 DWS tables in DuckDB from Lakehouse Parquet files."""
    cfg = settings or get_settings()
    cfg.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    con = duckdb.connect(str(cfg.duckdb_path))
    try:
        for table_name, hive_part in STAGE2_TABLES:
            table_dir = Path(cfg.lakehouse_dir / "dws" / table_name)
            if not table_dir.exists() or not any(table_dir.rglob("*.parquet")):
                continue
            glob_posix = f"{table_dir.resolve().as_posix()}/**/*.parquet"
            hive_flag = "true" if hive_part else "false"
            con.execute(
                f"""
                CREATE OR REPLACE TABLE {table_name} AS
                SELECT * FROM read_parquet('{glob_posix}', hive_partitioning={hive_flag})
                """
            )
            row = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
            counts[table_name] = int(row[0]) if row is not None else 0
    finally:
        con.close()

    return counts


def run_stage2_pipeline(
    settings: AppSettings | None = None,
    run_id: str | None = None,
) -> list[AuditRecord]:
    """Execute Stage 2 commercial diagnostics, rolling feature engineering, and DuckDB sync."""
    cfg = settings or get_settings()
    batch_run_id = run_id or generate_run_id("stage2")
    long_window = int(
        cfg.yaml_config.get("analytics", {}).get("windows", {}).get("long_window", 28)
    )

    spark = build_spark_session("LakePulse-Stage2-Analytics-Features", cfg)
    try:
        df_dwd = read_parquet(spark, cfg.resolve_table_uri("dwd", "dwd_daily_item_sales"))
        df_cal = read_parquet(spark, cfg.resolve_table_uri("ods", "ods_calendar"))
        dwd_rows = int(df_dwd.count())

        df_abc = build_abc_profile(df_dwd, cfg)
        rec_abc = _persist_dws_table(
            df_abc,
            cfg,
            batch_run_id,
            "dws_item_abc_profile",
            dwd_rows,
            ["store_id", "item_id", "abc_class", "safety_buffer_days"],
            message="ABC Pareto segmentation and safety buffer days computed.",
        )

        df_cal_fx = build_calendar_effect_analysis(df_dwd)
        rec_cal = _persist_dws_table(
            df_cal_fx,
            cfg,
            batch_run_id,
            "dws_calendar_effect_analysis",
            dwd_rows,
            ["store_id", "cat_id", "weekend_lift_pct", "holiday_lift_pct"],
            message="Weekend, holiday, SNAP coupon, and MoM lift metrics computed.",
        )

        df_price_fx = build_price_elasticity_analysis(df_dwd, long_window=long_window)
        rec_price = _persist_dws_table(
            df_price_fx,
            cfg,
            batch_run_id,
            "dws_price_elasticity_analysis",
            dwd_rows,
            ["store_id", "cat_id", "discount_bucket", "avg_item_sales_lift_pct"],
            message="Discount tier price elasticity metrics computed.",
        )

        df_features = build_rolling_features_df(spark, df_dwd, df_cal, cfg)
        rec_feat = _persist_dws_table(
            df_features,
            cfg,
            batch_run_id,
            "dws_item_rolling_features",
            dwd_rows,
            [
                "store_id",
                "item_id",
                "date",
                "sample_split",
                "sales_ma_7",
                "sales_ma_28",
                "next_7d_weekend_days",
            ],
            partition_cols=["store_id"],
            message="Leak-free rolling window features and embargo splits built.",
        )

        records = [rec_abc, rec_cal, rec_price, rec_feat]
        append_audit_records_spark(spark, cfg, records)
    finally:
        spark.stop()

    if cfg.mode == "local":
        sync_stage2_tables_to_duckdb(cfg)
        sync_audit_table_to_duckdb(cfg)

    return records
