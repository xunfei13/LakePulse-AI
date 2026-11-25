"""Distributed Spark @pandas_udf scoring and smart replenishment plan generation."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType

from lakepulse.config import AppSettings, get_settings
from lakepulse.etl.audit import (
    AuditRecord,
    append_audit_records_spark,
    generate_run_id,
    sync_audit_table_to_duckdb,
    utc_now_iso,
)
from lakepulse.ml.wide_and_deep import MODEL_INPUT_COLS, WideAndDeepPredictor
from lakepulse.spark.io import read_parquet, write_parquet
from lakepulse.spark.session import build_spark_session


def sync_ads_replenishment_to_duckdb(settings: AppSettings | None = None) -> int:
    """Sync ads_replenishment_plan Parquet table into DuckDB."""
    cfg = settings or get_settings()
    ads_dir = Path(cfg.lakehouse_dir / "ads" / "ads_replenishment_plan")
    if not ads_dir.exists() or not any(ads_dir.rglob("*.parquet")):
        return 0

    cfg.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    glob_posix = f"{ads_dir.resolve().as_posix()}/**/*.parquet"
    con = duckdb.connect(str(cfg.duckdb_path))
    try:
        con.execute(
            f"""
            CREATE OR REPLACE TABLE ads_replenishment_plan AS
            SELECT * FROM read_parquet('{glob_posix}')
            """
        )
        row = con.execute("SELECT COUNT(*) FROM ads_replenishment_plan").fetchone()
        return int(row[0]) if row is not None else 0
    finally:
        con.close()


def run_distributed_replenishment_inference(
    settings: AppSettings | None = None,
    run_id: str | None = None,
) -> AuditRecord:
    """Score active SKUs via broadcasted TensorFlow @pandas_udf and build replenishment plan."""
    cfg = settings or get_settings()
    batch_run_id = run_id or generate_run_id("infer")
    started_at = utc_now_iso()
    t0 = time.perf_counter()

    keras_path = cfg.models_dir / "wide_and_deep.keras"
    prep_path = cfg.models_dir / "preprocessor.json"
    if not keras_path.is_file() or not prep_path.is_file():
        raise FileNotFoundError(
            f"Missing trained Wide & Deep artifacts in {cfg.models_dir}. Run training first."
        )

    keras_bytes = keras_path.read_bytes()
    prep_dict = json.loads(prep_path.read_text(encoding="utf-8"))

    repl_cfg = cfg.yaml_config.get("replenishment", {})
    risk_cfg = repl_cfg.get("stockout_risk_coverage_days", {})
    high_thresh = float(risk_cfg.get("high_threshold", 3.0))
    med_thresh = float(risk_cfg.get("medium_threshold", 7.0))

    spark = build_spark_session("LakePulse-Stage3-Replenishment-Inference", cfg)
    try:
        bc_model_bundle = spark.sparkContext.broadcast(
            {"keras_bytes": keras_bytes, "preprocessor": prep_dict}
        )

        @pandas_udf(DoubleType())
        def predict_next_7d_udf(batch_iter: Iterator[pd.DataFrame]) -> Iterator[pd.Series]:
            bundle = bc_model_bundle.value
            predictor = WideAndDeepPredictor.from_serialized(
                bundle["keras_bytes"], bundle["preprocessor"]
            )
            for pdf in batch_iter:
                preds = predictor.predict_df(pdf)
                yield pd.Series(preds, index=pdf.index, dtype="float64")

        df_features = read_parquet(
            spark, cfg.resolve_table_uri("dws", "dws_item_rolling_features")
        )
        df_abc = read_parquet(
            spark, cfg.resolve_table_uri("dws", "dws_item_abc_profile")
        ).select("store_id", "item_id", "abc_class", "safety_buffer_days")

        df_infer = df_features.filter(F.col("sample_split") == "INFER").repartition("store_id")
        input_rows = int(df_infer.count())
        if input_rows == 0:
            raise RuntimeError("No INFER rows found in dws_item_rolling_features.")

        df_joined = df_infer.join(
            F.broadcast(df_abc), on=["store_id", "item_id"], how="left"
        ).fillna({"abc_class": "C", "safety_buffer_days": 1.0})

        df_scored = df_joined.withColumn(
            "pred_next_7d_sales",
            F.round(
                F.greatest(
                    F.lit(0.0),
                    predict_next_7d_udf(F.struct(*[F.col(c) for c in MODEL_INPUT_COLS])),
                ),
                2,
            ),
        ).cache()

        hash_unit = (
            F.pmod(F.abs(F.xxhash64(F.col("store_id"), F.col("item_id"))), F.lit(1000)).cast(
                "double"
            )
            / F.lit(1000.0)
        )
        stock_days_cover = F.lit(1.2) + hash_unit * F.lit(10.3)

        raw_tags = F.concat_ws(
            ",",
            F.when(F.col("abc_class") == "A", F.lit("CORE_A_SKU")),
            F.when(F.col("price_discount_rate") > 0.05, F.lit("PRICE_PROMO")),
            F.when(F.col("next_7d_event_days") > 0, F.lit("UPCOMING_HOLIDAY")),
            F.when(
                (F.col("cat_id") == "FOODS") & (F.col("next_7d_snap_days") > 0),
                F.lit("SNAP_VOUCHER_WINDOW"),
            ),
            F.when(F.col("sales_trend_ratio") > 1.15, F.lit("DEMAND_SURGE")),
        )

        df_plan = (
            df_scored.withColumn(
                "forecast_start_date", F.date_add(F.to_date(F.col("date")), 1)
            )
            .withColumn(
                "current_available_stock",
                F.round(
                    F.greatest(F.lit(0.0), F.col("sales_ma_28") * stock_days_cover), 0
                ).cast("int"),
            )
            .withColumn(
                "safety_stock_qty",
                F.round(
                    (F.col("pred_next_7d_sales") / F.lit(7.0)) * F.col("safety_buffer_days"),
                    2,
                ),
            )
            .withColumn(
                "recommended_order_qty",
                F.greatest(
                    F.lit(0),
                    F.ceil(
                        F.col("pred_next_7d_sales")
                        + F.col("safety_stock_qty")
                        - F.col("current_available_stock").cast("double")
                    ).cast("int"),
                ),
            )
            .withColumn(
                "coverage_days",
                F.round(
                    F.col("current_available_stock").cast("double")
                    / F.greatest(F.col("pred_next_7d_sales") / F.lit(7.0), F.lit(0.01)),
                    2,
                ),
            )
            .withColumn(
                "stockout_risk_level",
                F.when(F.col("coverage_days") < F.lit(high_thresh), F.lit("HIGH"))
                .when(F.col("coverage_days") < F.lit(med_thresh), F.lit("MEDIUM"))
                .otherwise(F.lit("LOW")),
            )
            .withColumn(
                "estimated_order_amount",
                F.round(F.col("recommended_order_qty").cast("double") * F.col("sell_price"), 2),
            )
            .withColumn(
                "key_driver_tags",
                F.when(raw_tags == "", F.lit("REGULAR_DEMAND")).otherwise(raw_tags),
            )
            .withColumn("model_version", F.lit("tensorflow_wide_and_deep"))
            .select(
                "store_id",
                "state_id",
                "cat_id",
                "dept_id",
                "item_id",
                "abc_class",
                "forecast_start_date",
                "sell_price",
                "sales_ma_7",
                "sales_ma_28",
                "pred_next_7d_sales",
                "safety_buffer_days",
                "safety_stock_qty",
                "current_available_stock",
                "recommended_order_qty",
                "estimated_order_amount",
                "coverage_days",
                "stockout_risk_level",
                "key_driver_tags",
                "model_version",
            )
        )

        output_rows = int(df_plan.count())
        shift_row = df_plan.agg(
            F.avg("pred_next_7d_sales").alias("avg_pred"),
            F.avg(F.col("sales_ma_28") * F.lit(7.0)).alias("avg_hist_7d"),
        ).collect()[0]
        avg_pred = float(shift_row["avg_pred"] or 0.0)
        avg_hist_7d = max(float(shift_row["avg_hist_7d"] or 0.0), 0.01)
        shift_ratio = round(avg_pred / avg_hist_7d, 4)

        write_parquet(
            df_plan,
            cfg.resolve_table_uri("ads", "ads_replenishment_plan"),
            mode="overwrite",
        )
        df_scored.unpersist()
        bc_model_bundle.unpersist()

        duration_sec = round(time.perf_counter() - t0, 3)
        finished_at = utc_now_iso()
        status = "SUCCESS" if 0.5 <= shift_ratio <= 2.0 else "WARN"
        audit_record = AuditRecord(
            run_id=batch_run_id,
            pipeline_stage="stage3_distributed_inference",
            table_name="ads_replenishment_plan",
            started_at=started_at,
            finished_at=finished_at,
            duration_sec=duration_sec,
            input_rows=input_rows,
            output_rows=output_rows,
            null_rate=0.0,
            status=status,
            message=(
                f"Scored {output_rows} SKUs via @pandas_udf; "
                f"avg_pred_7d={avg_pred:.2f}, shift_ratio={shift_ratio:.4f}."
            ),
        )
        append_audit_records_spark(spark, cfg, [audit_record])
    finally:
        spark.stop()

    if cfg.mode == "local":
        sync_ads_replenishment_to_duckdb(cfg)
        sync_audit_table_to_duckdb(cfg)

    return audit_record
