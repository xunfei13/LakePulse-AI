"""CLI entrypoint for Stage 3 multi-model training and time-split evaluation."""

from __future__ import annotations

import argparse

from lakepulse.config import get_settings
from lakepulse.ml.trainer import run_model_training_pipeline


def main() -> int:
    """Train Moving Avg, LightGBM, and TensorFlow Wide & Deep models and report metrics."""
    parser = argparse.ArgumentParser(description="Run LakePulse-AI Stage 3 model training.")
    parser.add_argument("--epochs", type=int, default=30, help="Max Keras training epochs.")
    args = parser.parse_args()

    cfg = get_settings()
    eval_rows, audit_rec = run_model_training_pipeline(cfg, epochs=args.epochs)

    print(
        f"[Stage 3] Training audit: status={audit_rec.status} "
        f"dur={audit_rec.duration_sec}s | {audit_rec.message}"
    )
    print("\n[Stage 3] Model Evaluation Leaderboard (Time-Based Splits):")
    print(
        f"  {'Model':<26} {'Split':<6} {'N':>6} {'MAE':>8} "
        f"{'RMSE':>8} {'WAPE%':>8} {'R2':>8} {'Tag'}"
    )
    for row in sorted(eval_rows, key=lambda r: (r["eval_split"], r["wape_pct"])):
        tags = []
        if row["is_best_test_wape"] and row["eval_split"] == "TEST":
            tags.append("BEST_TEST")
        if row["deployed_for_inference"]:
            tags.append("DEPLOYED")
        tag_str = ",".join(tags)
        print(
            f"  {row['model_name']:<26} {row['eval_split']:<6} "
            f"{row['sample_count']:>6d} {row['mae']:>8.3f} "
            f"{row['rmse']:>8.3f} {row['wape_pct']:>7.2f}% "
            f"{row['r2_score']:>8.3f} {tag_str}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
