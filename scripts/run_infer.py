"""CLI entrypoint for Stage 3 distributed @pandas_udf replenishment inference."""

from __future__ import annotations

import duckdb

from lakepulse.config import get_settings
from lakepulse.ml.replenishment_infer import run_distributed_replenishment_inference


def main() -> int:
    """Run distributed TensorFlow inference and print replenishment action summary."""
    cfg = get_settings()
    audit_rec = run_distributed_replenishment_inference(cfg)
    print(
        f"[Stage 3] Inference audit: status={audit_rec.status} "
        f"in={audit_rec.input_rows} out={audit_rec.output_rows} "
        f"dur={audit_rec.duration_sec}s | {audit_rec.message}"
    )

    con = duckdb.connect(str(cfg.duckdb_path), read_only=True)
    try:
        risk_summary = con.execute(
            """
            SELECT
                stockout_risk_level,
                COUNT(*) AS sku_cnt,
                SUM(recommended_order_qty) AS total_order_qty,
                ROUND(SUM(estimated_order_amount), 2) AS total_order_amount
            FROM ads_replenishment_plan
            GROUP BY stockout_risk_level
            ORDER BY CASE stockout_risk_level
                WHEN 'HIGH' THEN 1
                WHEN 'MEDIUM' THEN 2
                ELSE 3
            END
            """
        ).fetchall()

        top_orders = con.execute(
            """
            SELECT
                store_id,
                item_id,
                abc_class,
                pred_next_7d_sales,
                current_available_stock,
                recommended_order_qty,
                coverage_days,
                stockout_risk_level,
                key_driver_tags
            FROM ads_replenishment_plan
            WHERE recommended_order_qty > 0
            ORDER BY
                CASE stockout_risk_level WHEN 'HIGH' THEN 1 WHEN 'MEDIUM' THEN 2 ELSE 3 END,
                CASE abc_class WHEN 'A' THEN 1 WHEN 'B' THEN 2 ELSE 3 END,
                recommended_order_qty DESC
            LIMIT 10
            """
        ).fetchall()
    finally:
        con.close()

    print("\n[Stage 3] Replenishment Risk Summary:")
    for row in risk_summary:
        print(
            f"  * {row[0]:<6}: skus={row[1]:>3d}, order_qty={row[2]:>6d}, "
            f"est_amount=${row[3]:,.2f}"
        )

    print("\n[Stage 3] Top 10 Priority Replenishment Actions:")
    for row in top_orders:
        print(
            f"  * {row[0]} | {row[1]} (Class {row[2]}, {row[8]}): "
            f"pred_7d={row[3]:.1f}, stock={row[4]}, order={row[5]}, "
            f"cover={row[6]:.1f}d [{row[7]}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
