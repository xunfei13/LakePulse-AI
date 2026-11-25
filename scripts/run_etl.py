"""CLI entrypoint for Stage 1 M5 raw bootstrap and PySpark ODS ETL."""

from __future__ import annotations

import argparse

from lakepulse.config import get_settings
from lakepulse.etl.m5_bootstrap import ensure_m5_raw_data
from lakepulse.etl.spark_ods_etl import run_spark_ods_etl


def main() -> int:
    """Ensure raw M5 files and run distributed Spark ODS ETL."""
    parser = argparse.ArgumentParser(description="Run LakePulse-AI Stage 1 ODS ETL.")
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Regenerate synthetic M5 CSV files even if raw files already exist.",
    )
    args = parser.parse_args()

    cfg = get_settings()
    raw_info = ensure_m5_raw_data(cfg, force_regenerate=args.force_regenerate)
    print(f"[Stage 1] Raw M5 data source={raw_info['source']} dir={raw_info['raw_dir']}")

    records = run_spark_ods_etl(cfg)
    print("[Stage 1] PySpark ODS ETL audit summary:")
    for rec in records:
        print(
            f"  - {rec.table_name}: status={rec.status} "
            f"in={rec.input_rows} out={rec.output_rows} "
            f"null_rate={rec.null_rate} dur={rec.duration_sec}s | {rec.message}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
