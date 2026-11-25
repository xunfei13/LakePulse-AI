"""Stage 3 tests for Wide & Deep preprocessor, multi-model training, and @pandas_udf scoring."""

from __future__ import annotations

import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from lakepulse.analytics.pipeline import run_stage2_pipeline
from lakepulse.config import AppSettings, get_settings, load_yaml_settings
from lakepulse.etl.dbt_runner import run_dbt_build_and_publish
from lakepulse.etl.m5_bootstrap import ensure_m5_raw_data
from lakepulse.etl.spark_ods_etl import run_spark_ods_etl
from lakepulse.ml.replenishment_infer import run_distributed_replenishment_inference
from lakepulse.ml.trainer import run_model_training_pipeline
from lakepulse.ml.wide_and_deep import NUMERIC_COLS, TabularPreprocessor


def _make_tmp_settings(tmp_path: Path) -> AppSettings:
    """Create isolated local AppSettings pointing data and models into tmp_path."""
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


def _synthetic_feature_frame(
    store_ids: list[str],
    cat_ids: list[str],
    dept_ids: list[str],
    base_val: float,
) -> pd.DataFrame:
    """Build a small tabular feature DataFrame for preprocessor unit tests."""
    rows = []
    for idx, (s, c, d) in enumerate(zip(store_ids, cat_ids, dept_ids, strict=True)):
        row = {"store_id": s, "cat_id": c, "dept_id": d}
        for col_idx, col in enumerate(NUMERIC_COLS):
            row[col] = base_val + idx * 0.5 + col_idx * 0.1
        rows.append(row)
    return pd.DataFrame(rows)


def test_wide_and_deep_preprocessor_unseen_categories_and_no_leakage() -> None:
    """Verify TabularPreprocessor only fits on TRAIN and safely handles unseen categories."""
    train_df = _synthetic_feature_frame(
        ["CA_1", "CA_2", "CA_1"],
        ["FOODS", "HOBBIES", "FOODS"],
        ["FOODS_1", "HOBBIES_1", "FOODS_1"],
        base_val=2.0,
    )
    prep = TabularPreprocessor().fit(train_df)
    snapshot_means = dict(prep.means)
    snapshot_stds = dict(prep.stds)

    test_df = _synthetic_feature_frame(
        ["WI_9", "CA_1"],
        ["NEW_CAT", "FOODS"],
        ["NEW_DEPT", "FOODS_1"],
        base_val=500.0,
    )
    x_wide, x_deep = prep.transform(test_df)

    assert prep.means == snapshot_means
    assert prep.stds == snapshot_stds
    assert x_wide.shape == (2, prep.wide_dim)
    assert x_deep.shape == (2, prep.deep_dim)
    assert x_wide.dtype == np.float32 and x_deep.dtype == np.float32
    assert np.isfinite(x_wide).all() and np.isfinite(x_deep).all()

    # First row has unseen store/cat/dept -> all One-Hot columns in Deep branch must be 0
    ohe_slice = x_deep[0, len(NUMERIC_COLS) :]
    assert np.all(ohe_slice == 0.0)


def test_stage3_end_to_end_training_and_pandas_udf_replenishment(
    tmp_path: Path,
) -> None:
    """Verify Stage 3 training, model persistence, @pandas_udf scoring, and plan math."""
    cfg = _make_tmp_settings(tmp_path)
    ensure_m5_raw_data(
        cfg,
        num_days=90,
        items_per_store=9,
        inject_dirty=True,
        seed=42,
    )
    run_spark_ods_etl(cfg, run_id="s3_test_ods")
    run_dbt_build_and_publish(cfg, run_id="s3_test_dbt")
    run_stage2_pipeline(cfg, run_id="s3_test_s2")

    eval_rows, train_audit = run_model_training_pipeline(
        cfg,
        run_id="s3_test_train",
        epochs=12,
        seed=42,
    )
    assert len(eval_rows) == 6
    assert train_audit.status == "SUCCESS"
    assert (cfg.models_dir / "wide_and_deep.keras").is_file()
    assert (cfg.models_dir / "preprocessor.json").is_file()
    assert (cfg.models_dir / "metadata.json").is_file()
    assert (cfg.models_dir / "lightgbm_baseline.joblib").is_file()

    infer_audit = run_distributed_replenishment_inference(cfg, run_id="s3_test_infer")
    assert infer_audit.output_rows == 18

    con = duckdb.connect(str(cfg.duckdb_path), read_only=True)
    try:
        eval_cnt = con.execute("SELECT COUNT(*) FROM ops_model_eval_metrics").fetchone()[0]
        assert eval_cnt == 6

        metrics = con.execute(
            "SELECT mae, rmse, wape_pct FROM ops_model_eval_metrics"
        ).fetchall()
        for mae, rmse, wape in metrics:
            assert math.isfinite(mae) and mae >= 0.0
            assert math.isfinite(rmse) and rmse >= 0.0
            assert math.isfinite(wape) and wape >= 0.0

        plan_rows = con.execute(
            """
            SELECT
                pred_next_7d_sales,
                safety_stock_qty,
                current_available_stock,
                recommended_order_qty,
                stockout_risk_level,
                key_driver_tags
            FROM ads_replenishment_plan
            """
        ).fetchall()
        assert len(plan_rows) == 18

        for pred_7d, safety_qty, avail_stock, order_qty, risk, tags in plan_rows:
            assert pred_7d >= 0.0
            expected_qty = max(
                0,
                math.ceil(round(float(pred_7d) + float(safety_qty) - float(avail_stock), 6)),
            )
            assert int(order_qty) == expected_qty
            assert risk in {"HIGH", "MEDIUM", "LOW"}
            assert isinstance(tags, str) and len(tags) > 0

        s3_stages = {
            r[0]
            for r in con.execute(
                """
                SELECT DISTINCT pipeline_stage
                FROM ops_pipeline_audit
                WHERE pipeline_stage LIKE 'stage3_%'
                """
            ).fetchall()
        }
        assert s3_stages == {"stage3_model_training", "stage3_distributed_inference"}
    finally:
        con.close()
