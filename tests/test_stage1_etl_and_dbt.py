"""Stage 1 integration tests for M5 bootstrap, Spark ODS ETL, dbt quality gates, and audit."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

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


def test_m5_bootstrap_generates_expected_schema_and_anomalies(tmp_path: Path) -> None:
    """Verify M5 bootstrap creates official CSV columns and controlled dirty records."""
    cfg = _make_tmp_settings(tmp_path)
    info = ensure_m5_raw_data(
        cfg,
        num_days=35,
        items_per_store=12,
        inject_dirty=True,
        seed=42,
    )

    assert info["source"] == "generated"
    cal_df = pd.read_csv(info["calendar_file"])
    prices_df = pd.read_csv(info["prices_file"])
    sales_df = pd.read_csv(info["sales_file"])

    assert {"date", "wm_yr_wk", "weekday", "d", "snap_CA", "snap_TX", "snap_WI"}.issubset(
        cal_df.columns
    )
    assert len(cal_df) == 35
    assert list(prices_df.columns) == ["store_id", "item_id", "wm_yr_wk", "sell_price"]
    assert "d_1" in sales_df.columns and "d_35" in sales_df.columns
    assert set(sales_df["store_id"].unique()) == {"CA_1", "CA_2", "TX_1"}

    day_vals = sales_df[[f"d_{i}" for i in range(1, 36)]].to_numpy()
    assert (day_vals < 0).sum() == info["manifest"]["injected_negative_cells"]
    assert info["manifest"]["injected_negative_cells"] > 0
    assert prices_df.duplicated(subset=["store_id", "item_id", "wm_yr_wk"]).sum() > 0


def test_spark_ods_etl_and_dbt_end_to_end_pipeline(tmp_path: Path) -> None:
    """Verify full Stage 1 flow: Spark unpivot/dedup -> dbt cleaning/gates -> Parquet & audit."""
    cfg = _make_tmp_settings(tmp_path)
    ensure_m5_raw_data(
        cfg,
        num_days=35,
        items_per_store=9,
        inject_dirty=True,
        seed=7,
    )

    ods_records = run_spark_ods_etl(cfg, run_id="test_ods_run")
    assert {r.table_name for r in ods_records} == {
        "ods_calendar",
        "ods_sell_prices",
        "ods_sales_long",
    }
    ods_by_table = {r.table_name: r for r in ods_records}
    assert ods_by_table["ods_sell_prices"].status == "WARN"
    assert ods_by_table["ods_sales_long"].status == "WARN"

    dbt_records = run_dbt_build_and_publish(cfg, run_id="test_dbt_run")
    assert [r.table_name for r in dbt_records] == [
        "dwd_daily_item_sales",
        "dws_store_daily_summary",
    ]
    assert all(r.status == "SUCCESS" for r in dbt_records)

    con = duckdb.connect(str(cfg.duckdb_path), read_only=True)
    try:
        neg_cnt = con.execute(
            "SELECT COUNT(*) FROM dwd_daily_item_sales WHERE sales < 0 OR sell_price <= 0"
        ).fetchone()[0]
        repaired_cnt = con.execute(
            "SELECT SUM(is_repaired_negative) FROM dwd_daily_item_sales"
        ).fetchone()[0]
        dup_cnt = con.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT store_id, item_id, date
                FROM dwd_daily_item_sales
                GROUP BY store_id, item_id, date
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
        stores = {
            r[0]
            for r in con.execute("SELECT DISTINCT store_id FROM dwd_daily_item_sales").fetchall()
        }
        dwd_count = con.execute("SELECT COUNT(*) FROM dwd_daily_item_sales").fetchone()[0]
        audit_stages = {
            r[0]
            for r in con.execute(
                "SELECT DISTINCT pipeline_stage FROM ops_pipeline_audit"
            ).fetchall()
        }
    finally:
        con.close()

    assert neg_cnt == 0
    assert repaired_cnt > 0
    assert dup_cnt == 0
    assert stores == {"CA_1", "CA_2"}
    assert audit_stages == {"stage1_ods_etl", "stage1_dbt_build"}

    spark = build_spark_session("LakePulse-Stage1-Verify", cfg)
    try:
        df_dwd_spark = read_parquet(spark, cfg.resolve_table_uri("dwd", "dwd_daily_item_sales"))
        assert df_dwd_spark.count() == dwd_count
    finally:
        spark.stop()


def test_dbt_quality_gate_circuit_breaker_blocks_corrupted_data(tmp_path: Path) -> None:
    """Verify dbt build halts publication and records FAILED audit on duplicate calendar keys."""
    cfg = _make_tmp_settings(tmp_path)
    ensure_m5_raw_data(
        cfg,
        num_days=21,
        items_per_store=6,
        inject_dirty=False,
        seed=11,
    )
    run_spark_ods_etl(cfg, run_id="breaker_ods_run")

    # Corrupt ods_calendar Parquet by duplicating rows to violate dbt unique constraint on d/date
    cal_dir = Path(cfg.resolve_table_uri("ods", "ods_calendar"))
    cal_files = sorted(cal_dir.rglob("*.parquet"))
    cal_table = pq.read_table(cal_files[0])
    corrupted = pa.concat_tables([cal_table, cal_table])
    pq.write_table(corrupted, cal_files[0])

    with pytest.raises(RuntimeError, match="circuit breaker"):
        run_dbt_build_and_publish(cfg, run_id="breaker_dbt_run")

    dwd_dir = Path(cfg.resolve_table_uri("dwd", "dwd_daily_item_sales"))
    assert not dwd_dir.exists()

    con = duckdb.connect(str(cfg.duckdb_path), read_only=True)
    try:
        failed_rows = con.execute(
            """
            SELECT status, pipeline_stage
            FROM ops_pipeline_audit
            WHERE run_id = 'breaker_dbt_run'
            """
        ).fetchall()
    finally:
        con.close()

    assert failed_rows == [("FAILED", "stage1_dbt_build")]
