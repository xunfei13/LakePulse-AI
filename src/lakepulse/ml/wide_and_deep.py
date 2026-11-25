"""TensorFlow Keras Wide & Deep regression model and leak-free tabular preprocessor."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lakepulse.analytics.feature_engineering import FEATURE_COLUMNS

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

CATEGORICAL_COLS: tuple[str, ...] = ("store_id", "cat_id", "dept_id")
NUMERIC_COLS: tuple[str, ...] = (*FEATURE_COLUMNS, "wday", "month")
MODEL_INPUT_COLS: tuple[str, ...] = (*CATEGORICAL_COLS, *NUMERIC_COLS)
WIDE_NUMERIC_COUNT = 7


def _encode_one_hot(series: pd.Series, categories: list[str]) -> np.ndarray:
    """One-hot encode a string series against fixed categories (unseen -> all zeros)."""
    out = np.zeros((len(series), len(categories)), dtype=np.float32)
    if not categories or len(series) == 0:
        return out
    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    values = series.astype(str).tolist()
    for row_idx, val in enumerate(values):
        col_idx = cat_to_idx.get(val)
        if col_idx is not None:
            out[row_idx, col_idx] = 1.0
    return out


class TabularPreprocessor:
    """Fit-on-TRAIN-only preprocessor producing Wide and Deep float32 feature matrices."""

    def __init__(
        self,
        means: dict[str, float] | None = None,
        stds: dict[str, float] | None = None,
        vocab: dict[str, list[str]] | None = None,
    ) -> None:
        self.means: dict[str, float] = means or {}
        self.stds: dict[str, float] = stds or {}
        self.vocab: dict[str, list[str]] = vocab or {}
        self._is_fitted = bool(self.means and self.stds and self.vocab)

    @property
    def wide_dim(self) -> int:
        """Dimensionality of Wide branch input tensor."""
        if not self._is_fitted:
            raise RuntimeError("TabularPreprocessor must be fitted before accessing wide_dim.")
        return (
            WIDE_NUMERIC_COUNT
            + len(self.vocab["store_id"])
            + len(self.vocab["cat_id"])
        )

    @property
    def deep_dim(self) -> int:
        """Dimensionality of Deep branch input tensor."""
        if not self._is_fitted:
            raise RuntimeError("TabularPreprocessor must be fitted before accessing deep_dim.")
        ohe_total = sum(len(self.vocab[c]) for c in CATEGORICAL_COLS)
        return len(NUMERIC_COLS) + ohe_total

    def fit(self, train_df: pd.DataFrame) -> TabularPreprocessor:
        """Fit numeric standardization stats and categorical vocabularies on TRAIN split only."""
        if train_df.empty:
            raise ValueError("Cannot fit TabularPreprocessor on an empty training DataFrame.")

        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        for col in NUMERIC_COLS:
            arr = pd.to_numeric(train_df[col], errors="coerce").fillna(0.0).to_numpy(
                dtype=np.float64
            )
            means[col] = float(np.mean(arr))
            stds[col] = max(float(np.std(arr)), 1e-6)

        vocab: dict[str, list[str]] = {}
        for col in CATEGORICAL_COLS:
            vocab[col] = sorted(train_df[col].astype(str).unique().tolist())

        self.means = means
        self.stds = stds
        self.vocab = vocab
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Transform DataFrame into (X_wide, X_deep) float32 arrays."""
        if not self._is_fitted:
            raise RuntimeError("TabularPreprocessor must be fitted before transform().")

        if df.empty:
            return (
                np.empty((0, self.wide_dim), dtype=np.float32),
                np.empty((0, self.deep_dim), dtype=np.float32),
            )

        ohe_store = _encode_one_hot(df["store_id"], self.vocab["store_id"])
        ohe_cat = _encode_one_hot(df["cat_id"], self.vocab["cat_id"])
        ohe_dept = _encode_one_hot(df["dept_id"], self.vocab["dept_id"])

        ma7_week = (
            pd.to_numeric(df["sales_ma_7"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
            * 7.0
        )
        ma28_week = (
            pd.to_numeric(df["sales_ma_28"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
            * 7.0
        )
        lag1_week = (
            pd.to_numeric(df["sales_lag_1"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
            * 7.0
        )
        lag7_week = (
            pd.to_numeric(df["sales_lag_7"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
            * 7.0
        )
        disc_rate = (
            pd.to_numeric(df["price_discount_rate"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        next_events = (
            pd.to_numeric(df["next_7d_event_days"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
            / 7.0
        )
        next_wknds = (
            pd.to_numeric(df["next_7d_weekend_days"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
            / 7.0
        )

        x_wide = np.hstack(
            [
                ma7_week[:, None],
                ma28_week[:, None],
                lag1_week[:, None],
                lag7_week[:, None],
                (ma28_week * disc_rate)[:, None],
                (ma28_week * next_events)[:, None],
                (ma28_week * next_wknds)[:, None],
                ohe_store,
                ohe_cat,
            ]
        ).astype(np.float32)

        scaled_cols = []
        for col in NUMERIC_COLS:
            raw = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(self.means[col])
                .to_numpy(dtype=np.float32)
            )
            scaled = (raw - np.float32(self.means[col])) / np.float32(self.stds[col])
            scaled_cols.append(scaled[:, None])

        x_deep = np.hstack([*scaled_cols, ohe_store, ohe_cat, ohe_dept]).astype(np.float32)
        return x_wide, x_deep

    def to_dict(self) -> dict[str, Any]:
        """Serialize fitted parameters into a JSON-compatible dictionary."""
        if not self._is_fitted:
            raise RuntimeError("Cannot serialize an unfitted TabularPreprocessor.")
        return {
            "means": self.means,
            "stds": self.stds,
            "vocab": self.vocab,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TabularPreprocessor:
        """Restore a fitted TabularPreprocessor from a serialized dictionary."""
        return cls(
            means={str(k): float(v) for k, v in payload["means"].items()},
            stds={str(k): float(v) for k, v in payload["stds"].items()},
            vocab={str(k): [str(x) for x in vals] for k, vals in payload["vocab"].items()},
        )


def build_wide_and_deep_model(
    wide_dim: int,
    deep_dim: int,
    learning_rate: float = 3e-3,
) -> Any:
    """Build and compile the Keras Wide & Deep demand regression model."""
    import tensorflow as tf

    wide_in = tf.keras.Input(shape=(wide_dim,), name="wide_input")
    wide_out = tf.keras.layers.Dense(1, use_bias=True, name="wide_linear")(wide_in)

    deep_in = tf.keras.Input(shape=(deep_dim,), name="deep_input")
    h = tf.keras.layers.Dense(64, activation="relu", name="deep_dense_1")(deep_in)
    h = tf.keras.layers.Dropout(0.1, name="deep_dropout_1")(h)
    h = tf.keras.layers.Dense(32, activation="relu", name="deep_dense_2")(h)
    h = tf.keras.layers.Dense(16, activation="relu", name="deep_dense_3")(h)
    deep_out = tf.keras.layers.Dense(1, name="deep_head")(h)

    merged = tf.keras.layers.Add(name="wide_deep_sum")([wide_out, deep_out])
    output = tf.keras.layers.ReLU(name="pred_next_7d_sales")(merged)

    model = tf.keras.Model(
        inputs=[wide_in, deep_in],
        outputs=output,
        name="lakepulse_wide_and_deep",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber(delta=5.0),
        metrics=["mae"],
    )
    return model


class WideAndDeepPredictor:
    """Lightweight predictor bundle for batch scoring inside Spark partitions."""

    def __init__(self, model: Any, preprocessor: TabularPreprocessor) -> None:
        self.model = model
        self.preprocessor = preprocessor

    @staticmethod
    def save_bundle(
        model: Any,
        preprocessor: TabularPreprocessor,
        models_dir: Path,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Path]:
        """Persist Keras .keras model, preprocessor JSON, and training metadata."""
        models_dir.mkdir(parents=True, exist_ok=True)
        keras_path = models_dir / "wide_and_deep.keras"
        prep_path = models_dir / "preprocessor.json"
        meta_path = models_dir / "metadata.json"

        model.save(str(keras_path))
        prep_path.write_text(json.dumps(preprocessor.to_dict(), indent=2), encoding="utf-8")
        meta_path.write_text(json.dumps(metadata or {}, indent=2), encoding="utf-8")
        return {
            "keras_path": keras_path,
            "preprocessor_path": prep_path,
            "metadata_path": meta_path,
        }

    @classmethod
    def from_serialized(
        cls,
        keras_bytes: bytes,
        preprocessor_dict: dict[str, Any],
    ) -> WideAndDeepPredictor:
        """Load Keras model from broadcasted bytes using an isolated temp directory."""
        import tensorflow as tf

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_file = Path(tmp_dir) / "wide_and_deep.keras"
            model_file.write_bytes(keras_bytes)
            model = tf.keras.models.load_model(str(model_file), compile=False)

        preprocessor = TabularPreprocessor.from_dict(preprocessor_dict)
        return cls(model=model, preprocessor=preprocessor)

    @classmethod
    def from_dir(cls, models_dir: Path) -> WideAndDeepPredictor:
        """Load predictor directly from saved model directory."""
        keras_bytes = (models_dir / "wide_and_deep.keras").read_bytes()
        prep_dict = json.loads((models_dir / "preprocessor.json").read_text(encoding="utf-8"))
        return cls.from_serialized(keras_bytes, prep_dict)

    def predict_df(self, pdf: pd.DataFrame) -> np.ndarray:
        """Run non-negative forward prediction over a pandas DataFrame."""
        if pdf.empty:
            return np.empty(0, dtype=np.float64)
        x_wide, x_deep = self.preprocessor.transform(pdf)
        preds = self.model([x_wide, x_deep], training=False).numpy().reshape(-1)
        return np.maximum(0.0, preds.astype(np.float64))
