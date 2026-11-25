"""CLI entrypoint for Stage 1 dbt build, quality gates, and DuckDB audit summary."""

from __future__ import annotations

import duckdb

from lakepulse.config import get_settings
from lakepulse.etl.dbt_runner import run_dbt_build_and_publish


def main() -> int:
    """Run dbt build/test, publish Parquet marts, and summarize DuckDB outputs."""
    cfg = get_settings()
    records = run_dbt_build_and_publish(cfg)
    print("[Stage 1] dbt build & publication audit summary:")
    for rec in records:
        print(
            f"  - {rec.table_name}: status={rec.status} "
            f"in={rec.input_rows} out={rec.output_rows} "
            f"null_rate={rec.null_rate} dur={rec.duration_sec}s | {rec.message}"
        )

    con = duckdb.connect(str(cfg.duckdb_path), read_only=True)
    try:
        dwd_cnt = con.execute("SELECT COUNT(*) FROM dwd_daily_item_sales").fetchone()[0]
        dws_cnt = con.execute("SELECT COUNT(*) FROM dws_store_daily_summary").fetchone()[0]
        audit_cnt = con.execute("SELECT COUNT(*) FROM ops_pipeline_audit").fetchone()[0]
        latest = con.execute(
            """
            SELECT pipeline_stage, table_name, status, output_rows, duration_sec
            FROM ops_pipeline_audit
            ORDER BY finished_at DESC
            LIMIT 6
            """
        ).fetchall()
    finally:
        con.close()

    print(
        f"[Stage 1] DuckDB row counts -> dwd_daily_item_sales={dwd_cnt}, "
        f"dws_store_daily_summary={dws_cnt}, ops_pipeline_audit={audit_cnt}"
    )
    print("[Stage 1] Latest ops_pipeline_audit entries:")
    for row in latest:
        print(f"  * stage={row[0]} table={row[1]} status={row[2]} rows={row[3]} dur={row[4]}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
