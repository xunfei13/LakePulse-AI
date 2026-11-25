"""Stage 2 tests for commercial diagnostics, rolling features, and no-leakage invariants."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import pytest
from pyspark.sql import functions as F

from lakepulse.analytics.feature_engineering import (
    FEATURE_COLUMNS,
    build_rolling_features_df,
)
from lakepulse.analytics.pipeline import run_stage2_pipeline
from lakepulse.config import AppSettings, get_settings, load_yaml_settings
from lakepulse.etl.dbt_runner import run_dbt_build_and_publish
from lakepulse.etl.m5_bootstrap import ensure_m5_raw_data
from lakepulse.etl.spark_ods_etl import run_spark_ods_etl
from lakepulse.spark.io import read_parquet
from lakepulse.spark.session import build_spark_session


def _make_tmp_settings(tmp_path: Path) -> AppSettings:
    """Create isolated local AppSettings pointing data and DuckDB into tmp_path."""
    base_cfg = get_settings(mode="local")
    yaml_cfg = load_yaml_settings(base_cfg.project_root / "configs" / "settings.yaml")
    cfg = AppSettings(
        mode="local",
        project_root=base_cfg.project_root,
        data_dir_raw=tmp_path / "data",
        duckdb_path_raw=tmp_path / "data" / "lakehouse" / "lakepulse.duckdb",
        models_dir_raw=tmp_path / "artifacts" / "models",
        yaml_config=yaml_cfg,
    )
    assert tmp_path.resolve() in cfg.data_dir.parents
    return cfg


def test_stage2_abc_and_diagnostic_tables(tmp_path: Path) -> None:
    """Verify Stage 2 produces valid ABC segmentation, calendar/price lifts, and audit logs."""
    cfg = _make_tmp_settings(tmp_path)
    ensure_m5_raw_data(
        cfg,
        num_days=90,
        items_per_store=12,
        inject_dirty=True,
        seed=42,
    )
    run_spark_ods_etl(cfg, run_id="s2_test_ods")
    run_dbt_build_and_publish(cfg, run_id="s2_test_dbt")
    records = run_stage2_pipeline(cfg, run_id="s2_test_analytics")

    assert [r.table_name for r in records] == [
        "dws_item_abc_profile",
        "dws_calendar_effect_analysis",
        "dws_price_elasticity_analysis",
        "dws_item_rolling_features",
    ]
    assert all(r.status == "SUCCESS" and r.output_rows > 0 for r in records)

    con = duckdb.connect(str(cfg.duckdb_path), read_only=True)
    try:
        share_rows = con.execute(
            """
            SELECT store_id, SUM(store_revenue_share) AS share_sum
            FROM dws_item_abc_profile
            GROUP BY store_id
            """
        ).fetchall()
        assert len(share_rows) == 2
        for _, share_sum in share_rows:
            assert float(share_sum) == pytest.approx(1.0, abs=1e-3)

        buf_map = {
            r[0]: float(r[1])
            for r in con.execute(
                "SELECT DISTINCT abc_class, safety_buffer_days FROM dws_item_abc_profile"
            ).fetchall()
        }
        assert set(buf_map.keys()).issubset({"A", "B", "C"})
        expected_buffers = {"A": 3.0, "B": 2.0, "C": 1.0}
        for cls, buf in buf_map.items():
            assert buf == expected_buffers[cls]

        avg_lifts = con.execute(
            """
            SELECT AVG(weekend_lift_pct), AVG(holiday_lift_pct)
            FROM dws_calendar_effect_analysis
            """
        ).fetchone()
        assert avg_lifts is not None
        assert float(avg_lifts[0]) > 0.0
        assert float(avg_lifts[1]) > 0.0

        deep_promo = con.execute(
            """
            SELECT AVG(avg_item_sales_lift_pct)
            FROM dws_price_elasticity_analysis
            WHERE discount_bucket = '2_DEEP_PROMO_GT_10PCT'
            """
        ).fetchone()
        assert deep_promo is not None and deep_promo[0] is not None
        assert float(deep_promo[0]) > 0.0

        s2_audit_cnt = con.execute(
            """
            SELECT COUNT(*) FROM ops_pipeline_audit
            WHERE run_id = 's2_test_analytics' AND status = 'SUCCESS'
            """
        ).fetchone()[0]
        assert s2_audit_cnt == 4
    finally:
        con.close()


def test_rolling_features_perturbation_invariance_no_leakage(tmp_path: Path) -> None:
    """Prove features at date <= T0 are strictly invariant to future sales/price corruption."""
    cfg = _make_tmp_settings(tmp_path)
    ensure_m5_raw_data(
        cfg,
        num_days=90,
        items_per_store=6,
        inject_dirty=False,
        seed=99,
    )
    run_spark_ods_etl(cfg, run_id="leak_test_ods")
    run_dbt_build_and_publish(cfg, run_id="leak_test_dbt")

    spark = build_spark_session("LakePulse-Leakage-Test", cfg)
    try:
        df_dwd_clean = read_parquet(
            spark, cfg.resolve_table_uri("dwd", "dwd_daily_item_sales")
        ).withColumn("date", F.to_date(F.col("date")))
        df_cal = read_parquet(spark, cfg.resolve_table_uri("ods", "ods_calendar"))

        bounds = df_dwd_clean.agg(
            F.min("date").alias("min_d"), F.max("date").alias("max_d")
        ).collect()[0]
        min_date: date = bounds["min_d"]
        max_date: date = bounds["max_d"]
        cutoff_date = min_date + timedelta(days=50)

        df_dwd_perturbed = (
            df_dwd_clean.withColumn(
                "sales",
                F.when(F.col("date") > F.lit(cutoff_date), F.lit(99999)).otherwise(
                    F.col("sales")
                ),
            )
            .withColumn(
                "sell_price",
                F.when(F.col("date") > F.lit(cutoff_date), F.lit(999.99)).otherwise(
                    F.col("sell_price")
                ),
            )
            .withColumn(
                "revenue",
                F.when(
                    F.col("date") > F.lit(cutoff_date), F.lit(99999.0 * 999.99)
                ).otherwise(F.col("revenue")),
            )
        )

        f1_spark = build_rolling_features_df(spark, df_dwd_clean, df_cal, cfg)
        f2_spark = build_rolling_features_df(spark, df_dwd_perturbed, df_cal, cfg)

        compare_cols = [
            "store_id",
            "item_id",
            "date",
            "wday",
            "month",
            "sales",
            *FEATURE_COLUMNS,
        ]
        pdf1 = (
            f1_spark.filter(F.col("date") <= F.lit(cutoff_date))
            .select(*compare_cols)
            .orderBy("store_id", "item_id", "date")
            .toPandas()
        )
        pdf2 = (
            f2_spark.filter(F.col("date") <= F.lit(cutoff_date))
            .select(*compare_cols)
            .orderBy("store_id", "item_id", "date")
            .toPandas()
        )

        assert len(pdf1) > 0
        pd.testing.assert_frame_equal(pdf1, pdf2, check_dtype=True)

        split_pdf = (
            f1_spark.select("sample_split", "date", "target_next_7d_sales", *FEATURE_COLUMNS)
            .toPandas()
        )
        train_dates = split_pdf.loc[split_pdf["sample_split"] == "TRAIN", "date"]
        val_dates = split_pdf.loc[split_pdf["sample_split"] == "VAL", "date"]
        test_dates = split_pdf.loc[split_pdf["sample_split"] == "TEST", "date"]
        infer_df = split_pdf.loc[split_pdf["sample_split"] == "INFER"]

        assert len(train_dates) > 0
        assert len(val_dates) > 0
        assert len(test_dates) > 0
        assert len(infer_df) > 0

        assert train_dates.max() + timedelta(days=7) < val_dates.min()
        assert val_dates.max() + timedelta(days=7) < test_dates.min()
        assert set(infer_df["date"].unique()) == {max_date}
        assert infer_df["target_next_7d_sales"].isna().all()
        sched_cols = [
            "next_7d_weekend_days",
            "next_7d_event_days",
            "next_7d_snap_days",
        ]
        assert infer_df[sched_cols].notna().all().all()
    finally:
        spark.stop()
