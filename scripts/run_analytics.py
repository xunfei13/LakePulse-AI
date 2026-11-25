"""CLI entrypoint for Stage 2 commercial diagnostics and rolling feature engineering."""

from __future__ import annotations

import duckdb

from lakepulse.analytics.pipeline import run_stage2_pipeline
from lakepulse.config import get_settings


def main() -> int:
    """Run Stage 2 analytics/features pipeline and print commercial diagnostic summary."""
    cfg = get_settings()
    records = run_stage2_pipeline(cfg)
    print("[Stage 2] Pipeline audit summary:")
    for rec in records:
        print(
            f"  - {rec.table_name}: status={rec.status} "
            f"in={rec.input_rows} out={rec.output_rows} "
            f"null_rate={rec.null_rate} dur={rec.duration_sec}s | {rec.message}"
        )

    con = duckdb.connect(str(cfg.duckdb_path), read_only=True)
    try:
        abc_summary = con.execute(
            """
            SELECT
                abc_class,
                COUNT(*) AS sku_store_cnt,
                ROUND(SUM(total_revenue), 2) AS total_rev,
                ROUND(SUM(total_revenue) * 100.0 / SUM(SUM(total_revenue)) OVER (), 2) AS rev_pct,
                MAX(safety_buffer_days) AS buffer_days
            FROM dws_item_abc_profile
            GROUP BY abc_class
            ORDER BY abc_class
            """
        ).fetchall()

        cal_summary = con.execute(
            """
            SELECT
                store_id,
                cat_id,
                weekend_lift_pct,
                holiday_lift_pct,
                snap_lift_pct,
                latest_month_mom_pct
            FROM dws_calendar_effect_analysis
            ORDER BY store_id, cat_id
            """
        ).fetchall()

        price_summary = con.execute(
            """
            SELECT
                discount_bucket,
                SUM(obs_item_days) AS obs_days,
                ROUND(AVG(avg_discount_pct), 2) AS mean_disc_pct,
                ROUND(AVG(avg_item_sales_lift_pct), 2) AS mean_sales_lift_pct
            FROM dws_price_elasticity_analysis
            GROUP BY discount_bucket
            ORDER BY discount_bucket
            """
        ).fetchall()

        split_summary = con.execute(
            """
            SELECT
                sample_split,
                COUNT(*) AS row_cnt,
                MIN(date) AS min_date,
                MAX(date) AS max_date,
                COUNT(target_next_7d_sales) AS labeled_cnt
            FROM dws_item_rolling_features
            GROUP BY sample_split
            ORDER BY MIN(date)
            """
        ).fetchall()
    finally:
        con.close()

    print("\n[Stage 2] 1. ABC Value Segmentation Summary:")
    for row in abc_summary:
        print(
            f"  * Class {row[0]}: store_items={row[1]}, revenue={row[2]} "
            f"({row[3]}%), safety_buffer_days={row[4]}"
        )

    print("\n[Stage 2] 2. Calendar & Promotion Lift Analysis (%):")
    for row in cal_summary:
        print(
            f"  * {row[0]} | {row[1]}: weekend={row[2]:+.2f}%, "
            f"holiday={row[3]:+.2f}%, snap={row[4]:+.2f}%, latest_mom={row[5]:+.2f}%"
        )

    print("\n[Stage 2] 3. Price Discount Elasticity Summary:")
    for row in price_summary:
        print(
            f"  * {row[0]}: obs_days={row[1]}, avg_discount={row[2]:.2f}%, "
            f"avg_item_sales_lift={row[3]:+.2f}%"
        )

    print("\n[Stage 2] 4. Rolling Feature Time-Split & Embargo Ranges:")
    for row in split_summary:
        print(
            f"  * {row[0]:7s}: rows={row[1]:5d}, range=[{row[2]} .. {row[3]}], "
            f"labeled={row[4]}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
