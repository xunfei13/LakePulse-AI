"""Stage 3 multi-model training and time-split evaluation pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from lakepulse.config import AppSettings, get_settings
from lakepulse.etl.audit import (
    AuditRecord,
    append_audit_records_spark,
    generate_run_id,
    sync_audit_table_to_duckdb,
    utc_now_iso,
)
from lakepulse.ml.wide_and_deep import (
    CATEGORICAL_COLS,
    MODEL_INPUT_COLS,
    NUMERIC_COLS,
    TabularPreprocessor,
    WideAndDeepPredictor,
    build_wide_and_deep_model,
)
from lakepulse.spark.io import read_parquet, write_parquet
from lakepulse.spark.session import build_spark_session

EVAL_METRICS_SCHEMA = StructType(
    [
        StructField("run_id", StringType(), nullable=False),
        StructField("evaluated_at", StringType(), nullable=False),
        StructField("model_name", StringType(), nullable=False),
        StructField("eval_split", StringType(), nullable=False),
        StructField("sample_count", LongType(), nullable=False),
        StructField("mae", DoubleType(), nullable=False),
        StructField("rmse", DoubleType(), nullable=False),
        StructField("wape_pct", DoubleType(), nullable=False),
        StructField("r2_score", DoubleType(), nullable=False),
        StructField("is_best_test_wape", BooleanType(), nullable=False),
        StructField("deployed_for_inference", BooleanType(), nullable=False),
    ]
)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute MAE, RMSE, WAPE (%), and R2 regression metrics."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.maximum(0.0, np.asarray(y_pred, dtype=np.float64))

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    denom = max(float(np.sum(np.abs(y_t))), 1e-6)
    wape_pct = float(np.sum(np.abs(y_t - y_p)) / denom * 100.0)
    r2 = 0.0 if float(np.var(y_t)) < 1e-12 else float(r2_score(y_t, y_p))

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "wape_pct": round(wape_pct, 4),
        "r2_score": round(r2, 4),
    }


def _fit_lgbm_cat_maps(train_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Build deterministic category-to-integer mappings strictly from TRAIN split."""
    cat_maps: dict[str, dict[str, int]] = {}
    for col in CATEGORICAL_COLS:
        cats = sorted(train_df[col].astype(str).unique().tolist())
        cat_maps[col] = {val: idx for idx, val in enumerate(cats)}
    return cat_maps


def _transform_lgbm_features(
    df: pd.DataFrame,
    cat_maps: dict[str, dict[str, int]],
) -> np.ndarray:
    """Convert tabular DataFrame into float32 matrix for LightGBM."""
    cat_arrays = []
    for col in CATEGORICAL_COLS:
        mapping = cat_maps[col]
        codes = (
            df[col]
            .astype(str)
            .map(lambda v, m=mapping: m.get(v, -1))
            .to_numpy(dtype=np.float32)
        )
        cat_arrays.append(codes[:, None])

    num_arrays = [
        pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)[:, None]
        for col in NUMERIC_COLS
    ]
    return np.hstack([*cat_arrays, *num_arrays]).astype(np.float32)


def sync_eval_metrics_to_duckdb(settings: AppSettings | None = None) -> int:
    """Sync ops_model_eval_metrics Parquet table into DuckDB."""
    cfg = settings or get_settings()
    metrics_dir = Path(cfg.lakehouse_dir / "ops" / "ops_model_eval_metrics")
    if not metrics_dir.exists() or not any(metrics_dir.rglob("*.parquet")):
        return 0

    cfg.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    glob_posix = f"{metrics_dir.resolve().as_posix()}/**/*.parquet"
    con = duckdb.connect(str(cfg.duckdb_path))
    try:
        con.execute(
            f"""
            CREATE OR REPLACE TABLE ops_model_eval_metrics AS
            SELECT * FROM read_parquet('{glob_posix}')
            """
        )
        row = con.execute("SELECT COUNT(*) FROM ops_model_eval_metrics").fetchone()
        return int(row[0]) if row is not None else 0
    finally:
        con.close()


def run_model_training_pipeline(
    settings: AppSettings | None = None,
    run_id: str | None = None,
    *,
    epochs: int = 30,
    batch_size: int = 256,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], AuditRecord]:
    """Train Naive, LightGBM, and TensorFlow Wide & Deep models on strict time splits."""
    import tensorflow as tf

    cfg = settings or get_settings()
    batch_run_id = run_id or generate_run_id("train")
    started_at = utc_now_iso()
    t0 = time.perf_counter()

    spark = build_spark_session("LakePulse-Stage3-Model-Training", cfg)
    try:
        df_features = read_parquet(
            spark, cfg.resolve_table_uri("dws", "dws_item_rolling_features")
        )
        df_modeling = (
            df_features.filter(
                F.col("sample_split").isin("TRAIN", "VAL", "TEST")
                & F.col("target_next_7d_sales").isNotNull()
            )
            .select(
                "store_id",
                "item_id",
                "date",
                "sample_split",
                *[c for c in MODEL_INPUT_COLS if c != "store_id"],
                "target_next_7d_sales",
            )
            .toPandas()
        )
    finally:
        spark.stop()

    if df_modeling.empty:
        raise RuntimeError("No TRAIN/VAL/TEST rows found in dws_item_rolling_features.")

    df_modeling["date"] = pd.to_datetime(df_modeling["date"]).dt.date
    train_df = df_modeling[df_modeling["sample_split"] == "TRAIN"].reset_index(drop=True)
    val_df = df_modeling[df_modeling["sample_split"] == "VAL"].reset_index(drop=True)
    test_df = df_modeling[df_modeling["sample_split"] == "TEST"].reset_index(drop=True)

    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError(
            f"All splits must be non-empty: TRAIN={len(train_df)}, "
            f"VAL={len(val_df)}, TEST={len(test_df)}"
        )

    if not (train_df["date"].max() < val_df["date"].min() < test_df["date"].min()):
        raise RuntimeError("Time-based split ordering violated across TRAIN/VAL/TEST.")

    y_train = train_df["target_next_7d_sales"].to_numpy(dtype=np.float32)
    y_val = val_df["target_next_7d_sales"].to_numpy(dtype=np.float32)
    y_test = test_df["target_next_7d_sales"].to_numpy(dtype=np.float32)

    tf.keras.utils.set_random_seed(seed)

    # 1. Statistical baseline: trailing 7-day average scaled to 7-day horizon
    val_pred_naive = np.maximum(0.0, val_df["sales_ma_7"].to_numpy(dtype=np.float64) * 7.0)
    test_pred_naive = np.maximum(0.0, test_df["sales_ma_7"].to_numpy(dtype=np.float64) * 7.0)

    # 2. Gradient-boosted tree baseline: LightGBM
    cat_maps = _fit_lgbm_cat_maps(train_df)
    x_lgb_train = _transform_lgbm_features(train_df, cat_maps)
    x_lgb_val = _transform_lgbm_features(val_df, cat_maps)
    x_lgb_test = _transform_lgbm_features(test_df, cat_maps)

    lgbm_model = lgb.LGBMRegressor(
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        verbose=-1,
    )
    lgbm_model.fit(
        x_lgb_train,
        y_train,
        eval_set=[(x_lgb_val, y_val)],
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )
    val_pred_lgbm = np.maximum(0.0, lgbm_model.predict(x_lgb_val))
    test_pred_lgbm = np.maximum(0.0, lgbm_model.predict(x_lgb_test))

    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": lgbm_model, "cat_maps": cat_maps, "feature_cols": list(MODEL_INPUT_COLS)},
        cfg.models_dir / "lightgbm_baseline.joblib",
    )

    # 3. Core deep learning model: TensorFlow Wide & Deep
    preprocessor = TabularPreprocessor().fit(train_df)
    xw_train, xd_train = preprocessor.transform(train_df)
    xw_val, xd_val = preprocessor.transform(val_df)

    wd_model = build_wide_and_deep_model(
        wide_dim=preprocessor.wide_dim,
        deep_dim=preprocessor.deep_dim,
        learning_rate=3e-3,
    )
    eff_batch_size = min(batch_size, max(16, len(train_df) // 16))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=0
        ),
    ]
    wd_model.fit(
        [xw_train, xd_train],
        y_train,
        validation_data=([xw_val, xd_val], y_val),
        epochs=epochs,
        batch_size=eff_batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    predictor = WideAndDeepPredictor(model=wd_model, preprocessor=preprocessor)
    val_pred_tf = predictor.predict_df(val_df)
    test_pred_tf = predictor.predict_df(test_df)

    evaluated_at = utc_now_iso()
    raw_eval_rows: list[dict[str, Any]] = []
    model_preds = (
        ("moving_avg_baseline", val_pred_naive, test_pred_naive),
        ("lightgbm_baseline", val_pred_lgbm, test_pred_lgbm),
        ("tensorflow_wide_and_deep", val_pred_tf, test_pred_tf),
    )

    test_wapes: dict[str, float] = {}
    for model_name, vp, tp in model_preds:
        val_m = compute_regression_metrics(y_val, vp)
        test_m = compute_regression_metrics(y_test, tp)
        test_wapes[model_name] = test_m["wape_pct"]

        raw_eval_rows.append(
            {
                "run_id": batch_run_id,
                "evaluated_at": evaluated_at,
                "model_name": model_name,
                "eval_split": "VAL",
                "sample_count": int(len(val_df)),
                **val_m,
            }
        )
        raw_eval_rows.append(
            {
                "run_id": batch_run_id,
                "evaluated_at": evaluated_at,
                "model_name": model_name,
                "eval_split": "TEST",
                "sample_count": int(len(test_df)),
                **test_m,
            }
        )

    best_model_name = min(test_wapes, key=lambda name: test_wapes[name])
    eval_rows: list[dict[str, Any]] = []
    for row in raw_eval_rows:
        eval_rows.append(
            {
                **row,
                "is_best_test_wape": row["model_name"] == best_model_name,
                "deployed_for_inference": row["model_name"] == "tensorflow_wide_and_deep",
            }
        )

    WideAndDeepPredictor.save_bundle(
        wd_model,
        preprocessor,
        cfg.models_dir,
        metadata={
            "run_id": batch_run_id,
            "trained_at": evaluated_at,
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "best_test_wape_model": best_model_name,
            "test_wape_pct": test_wapes,
        },
    )

    duration_sec = round(time.perf_counter() - t0, 3)
    finished_at = utc_now_iso()
    audit_record = AuditRecord(
        run_id=batch_run_id,
        pipeline_stage="stage3_model_training",
        table_name="ops_model_eval_metrics",
        started_at=started_at,
        finished_at=finished_at,
        duration_sec=duration_sec,
        input_rows=int(len(df_modeling)),
        output_rows=len(eval_rows),
        null_rate=0.0,
        status="SUCCESS",
        message=(
            f"Trained 3 models; best_test_wape={best_model_name} "
            f"({test_wapes[best_model_name]:.2f}%), "
            f"tf_test_wape={test_wapes['tensorflow_wide_and_deep']:.2f}%."
        ),
    )

    spark_persist = build_spark_session("LakePulse-Stage3-Eval-Persist", cfg)
    try:
        df_metrics = spark_persist.createDataFrame(eval_rows, schema=EVAL_METRICS_SCHEMA)
        write_parquet(
            df_metrics,
            cfg.resolve_table_uri("ops", "ops_model_eval_metrics"),
            mode="overwrite",
        )
        append_audit_records_spark(spark_persist, cfg, [audit_record])
    finally:
        spark_persist.stop()

    if cfg.mode == "local":
        sync_eval_metrics_to_duckdb(cfg)
        sync_audit_table_to_duckdb(cfg)

    return eval_rows, audit_record
