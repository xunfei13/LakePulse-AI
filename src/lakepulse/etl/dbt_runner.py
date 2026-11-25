"""dbt-duckdb runner with quality-gate circuit breaker and Parquet publication."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import duckdb

from lakepulse.config import AppSettings, get_settings
from lakepulse.etl.audit import (
    AuditRecord,
    append_audit_records_local,
    generate_run_id,
    sync_audit_table_to_duckdb,
    utc_now_iso,
)


def _export_duckdb_table_to_parquet(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    target_dir: Path,
) -> None:
    """Atomically overwrite target_dir/part-00000.parquet from a DuckDB table."""
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = (target_dir / "part-00000.parquet").resolve().as_posix()
    con.execute(f"COPY (SELECT * FROM {table_name}) TO '{out_file}' (FORMAT PARQUET)")


def _measure_table_null_rate(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    cols: list[str],
) -> tuple[int, float]:
    """Return (row_count, avg_null_rate) for key columns of a DuckDB table."""
    row = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    row_count = int(row[0]) if row is not None else 0
    if row_count == 0 or not cols:
        return row_count, 0.0

    null_exprs = " + ".join(
        f"SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END)" for c in cols
    )
    null_row = con.execute(f"SELECT ({null_exprs}) FROM {table_name}").fetchone()
    total_nulls = int(null_row[0] or 0) if null_row is not None else 0
    return row_count, round(total_nulls / (row_count * len(cols)), 6)


def run_dbt_build_and_publish(
    settings: AppSettings | None = None,
    run_id: str | None = None,
) -> list[AuditRecord]:
    """Run dbt build (models + quality gates); publish DWD/DWS Parquet on pass or trip breaker."""
    cfg = settings or get_settings()
    batch_run_id = run_id or generate_run_id("dbt")
    dbt_dir = cfg.project_root / "dbt_project"

    cfg.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LAKEPULSE_DUCKDB_PATH"] = cfg.duckdb_path.resolve().as_posix()
    env["LAKEPULSE_LAKEHOUSE_DIR"] = cfg.lakehouse_dir.resolve().as_posix()

    started_at = utc_now_iso()
    t0 = time.perf_counter()

    cmd = [
        sys.executable,
        "-m",
        "dbt.cli.main",
        "build",
        "--project-dir",
        str(dbt_dir),
        "--profiles-dir",
        str(dbt_dir),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    duration_sec = round(time.perf_counter() - t0, 3)
    finished_at = utc_now_iso()

    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip(), file=sys.stderr)

    if proc.returncode != 0:
        combined = f"{proc.stdout}\n{proc.stderr}".strip()
        tail_lines = " | ".join(combined.splitlines()[-6:])[:500]
        failed_record = AuditRecord(
            run_id=batch_run_id,
            pipeline_stage="stage1_dbt_build",
            table_name="dwd_daily_item_sales",
            started_at=started_at,
            finished_at=finished_at,
            duration_sec=duration_sec,
            input_rows=0,
            output_rows=0,
            null_rate=1.0,
            status="FAILED",
            message=f"dbt quality gate failed (exit={proc.returncode}): {tail_lines}",
        )
        append_audit_records_local(cfg, [failed_record])
        try:
            sync_audit_table_to_duckdb(cfg)
        except Exception:
            pass
        raise RuntimeError(
            f"dbt build circuit breaker triggered (exit={proc.returncode}): {tail_lines}"
        )

    con = duckdb.connect(str(cfg.duckdb_path))
    try:
        ods_row = con.execute("SELECT COUNT(*) FROM stg_ods_sales_long").fetchone()
        ods_rows = int(ods_row[0]) if ods_row is not None else 0

        dwd_rows, dwd_null_rate = _measure_table_null_rate(
            con,
            "dwd_daily_item_sales",
            ["store_id", "item_id", "date", "sales", "sell_price", "revenue"],
        )
        dws_rows, dws_null_rate = _measure_table_null_rate(
            con,
            "dws_store_daily_summary",
            ["store_id", "date", "total_sales_qty", "total_revenue"],
        )
        repaired_row = con.execute(
            "SELECT COALESCE(SUM(is_repaired_negative), 0) FROM dwd_daily_item_sales"
        ).fetchone()
        repaired_count = int(repaired_row[0]) if repaired_row is not None else 0

        _export_duckdb_table_to_parquet(
            con,
            "dwd_daily_item_sales",
            cfg.lakehouse_dir / "dwd" / "dwd_daily_item_sales",
        )
        _export_duckdb_table_to_parquet(
            con,
            "dws_store_daily_summary",
            cfg.lakehouse_dir / "dws" / "dws_store_daily_summary",
        )
    finally:
        con.close()

    records = [
        AuditRecord(
            run_id=batch_run_id,
            pipeline_stage="stage1_dbt_build",
            table_name="dwd_daily_item_sales",
            started_at=started_at,
            finished_at=finished_at,
            duration_sec=duration_sec,
            input_rows=ods_rows,
            output_rows=dwd_rows,
            null_rate=dwd_null_rate,
            status="SUCCESS",
            message=(
                f"dbt build passed; published Parquet (repaired_negatives={repaired_count})."
            ),
        ),
        AuditRecord(
            run_id=batch_run_id,
            pipeline_stage="stage1_dbt_build",
            table_name="dws_store_daily_summary",
            started_at=started_at,
            finished_at=finished_at,
            duration_sec=duration_sec,
            input_rows=dwd_rows,
            output_rows=dws_rows,
            null_rate=dws_null_rate,
            status="SUCCESS",
            message="dbt build passed; published store daily summary Parquet.",
        ),
    ]
    append_audit_records_local(cfg, records)
    sync_audit_table_to_duckdb(cfg)
    return records
