"""Walmart M5 raw dataset detection and deterministic high-fidelity generator."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lakepulse.config import AppSettings, get_settings
from lakepulse.etl.audit import utc_now_iso

STORE_SPEC: tuple[tuple[str, str], ...] = (
    ("CA_1", "CA"),
    ("CA_2", "CA"),
    ("TX_1", "TX"),
)

DEPT_SPECS: tuple[tuple[str, str], ...] = (
    ("FOODS_1", "FOODS"),
    ("HOUSEHOLD_1", "HOUSEHOLD"),
    ("HOBBIES_1", "HOBBIES"),
)

HOLIDAY_EVENTS: dict[tuple[int, int], tuple[str, str]] = {
    (1, 1): ("NewYear", "National"),
    (2, 1): ("SuperBowl", "Sporting"),
    (2, 16): ("PresidentsDay", "National"),
    (5, 25): ("MemorialDay", "National"),
    (7, 4): ("IndependenceDay", "National"),
    (9, 7): ("LaborDay", "National"),
    (10, 31): ("Halloween", "Cultural"),
    (11, 11): ("VeteransDay", "National"),
    (11, 26): ("Thanksgiving", "National"),
    (12, 25): ("Christmas", "National"),
}


def _find_existing_m5_files(raw_dir: Path) -> dict[str, Path] | None:
    """Return official or bootstrapped M5 CSV paths if present in raw_dir."""
    calendar_path = raw_dir / "calendar.csv"
    prices_path = raw_dir / "sell_prices.csv"
    eval_sales = raw_dir / "sales_train_evaluation.csv"
    val_sales = raw_dir / "sales_train_validation.csv"

    sales_path = eval_sales if eval_sales.is_file() else val_sales
    if calendar_path.is_file() and prices_path.is_file() and sales_path.is_file():
        return {
            "calendar_file": calendar_path,
            "prices_file": prices_path,
            "sales_file": sales_path,
        }
    return None


def _build_calendar_df(num_days: int, start_date: date) -> pd.DataFrame:
    """Build M5-schema calendar dataframe covering num_days from start_date."""
    # M5 wday mapping: Saturday=1, Sunday=2, Monday=3, ..., Friday=7
    python_to_m5_wday = {5: 1, 6: 2, 0: 3, 1: 4, 2: 5, 3: 6, 4: 7}
    weekday_names = (
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    )

    records: list[dict[str, Any]] = []
    wm_week_idx = 11501
    for i in range(num_days):
        cur = start_date + timedelta(days=i)
        m5_wday = python_to_m5_wday[cur.weekday()]
        if i > 0 and m5_wday == 1:
            wm_week_idx += 1

        event_name, event_type = HOLIDAY_EVENTS.get((cur.month, cur.day), ("", ""))
        snap_active = 1 if cur.day <= 10 else 0
        records.append(
            {
                "date": cur.isoformat(),
                "wm_yr_wk": wm_week_idx,
                "weekday": weekday_names[cur.weekday()],
                "wday": m5_wday,
                "month": cur.month,
                "year": cur.year,
                "d": f"d_{i + 1}",
                "event_name_1": event_name,
                "event_type_1": event_type,
                "event_name_2": "",
                "event_type_2": "",
                "snap_CA": snap_active,
                "snap_TX": snap_active,
                "snap_WI": snap_active,
            }
        )
    return pd.DataFrame(records)


def _build_item_catalog(items_per_store: int) -> list[tuple[str, str, str]]:
    """Build deterministic item list (item_id, dept_id, cat_id) shared across stores."""
    counters = {dept: 0 for dept, _ in DEPT_SPECS}
    catalog: list[tuple[str, str, str]] = []
    for idx in range(items_per_store):
        dept_id, cat_id = DEPT_SPECS[idx % len(DEPT_SPECS)]
        counters[dept_id] += 1
        item_id = f"{dept_id}_{counters[dept_id]:03d}"
        catalog.append((item_id, dept_id, cat_id))
    return catalog


def ensure_m5_raw_data(
    settings: AppSettings | None = None,
    *,
    force_regenerate: bool = False,
    num_days: int = 365,
    items_per_store: int = 60,
    inject_dirty: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Ensure Walmart M5 CSV files exist in data/raw, generating realistic M5 data if absent."""
    if num_days < 7:
        raise ValueError("num_days must be at least 7.")
    if items_per_store < 3:
        raise ValueError("items_per_store must be at least 3.")

    cfg = settings or get_settings()
    raw_dir = cfg.data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    existing = _find_existing_m5_files(raw_dir)
    if existing is not None and not force_regenerate:
        return {
            "source": "existing",
            "raw_dir": str(raw_dir),
            "calendar_file": str(existing["calendar_file"]),
            "prices_file": str(existing["prices_file"]),
            "sales_file": str(existing["sales_file"]),
        }

    rng = np.random.default_rng(seed)
    start_date = date(2015, 1, 1)
    cal_df = _build_calendar_df(num_days=num_days, start_date=start_date)
    wm_weeks = cal_df["wm_yr_wk"].to_numpy()
    unique_weeks = cal_df["wm_yr_wk"].drop_duplicates().tolist()

    is_weekend = cal_df["weekday"].isin(["Saturday", "Sunday"]).to_numpy()
    is_event = (cal_df["event_name_1"] != "").to_numpy()
    snap_by_state = {
        "CA": cal_df["snap_CA"].to_numpy(),
        "TX": cal_df["snap_TX"].to_numpy(),
        "WI": cal_df["snap_WI"].to_numpy(),
    }

    catalog = _build_item_catalog(items_per_store)
    # Pareto weights so ~20% of items drive ~70% of revenue for ABC segmentation
    raw_pareto = np.sort(rng.pareto(a=1.35, size=items_per_store) + 0.35)[::-1]
    item_popularity = 1.2 + 9.0 * (raw_pareto / raw_pareto.max()) ** 1.35

    base_prices = np.round(rng.uniform(2.5, 24.5, size=items_per_store), 2)
    store_scale = {"CA_1": 1.15, "CA_2": 0.95, "TX_1": 1.00}

    price_rows: list[dict[str, Any]] = []
    sales_rows: list[dict[str, Any]] = []
    day_cols = [f"d_{i + 1}" for i in range(num_days)]

    injected_negative_cells = 0
    injected_duplicate_prices = 0

    for store_id, state_id in STORE_SPEC:
        s_scale = store_scale.get(store_id, 1.0)
        snap_vec = snap_by_state[state_id]

        for item_idx, (item_id, dept_id, cat_id) in enumerate(catalog):
            # ~15% of items launch later so pre-launch periods have null sell_price
            has_late_launch = (item_idx % 7 == 6) and (num_days >= 28)
            if has_late_launch:
                low_day = max(7, num_days // 10)
                high_day = max(14, num_days // 3)
                launch_day = int(rng.integers(low_day, high_day))
            else:
                launch_day = 0
            launch_week = int(wm_weeks[launch_day])

            base_price = float(base_prices[item_idx])
            week_price_map: dict[int, float] = {}
            for wk in unique_weeks:
                if wk < launch_week:
                    continue
                is_promo_week = rng.random() < 0.15
                discount = float(rng.uniform(0.75, 0.90)) if is_promo_week else 1.0
                sell_price = round(base_price * discount, 2)
                week_price_map[int(wk)] = sell_price
                price_rows.append(
                    {
                        "store_id": store_id,
                        "item_id": item_id,
                        "wm_yr_wk": int(wk),
                        "sell_price": sell_price,
                    }
                )

            if inject_dirty and item_idx == 0 and unique_weeks:
                dup_week = int(unique_weeks[-1])
                price_rows.append(
                    {
                        "store_id": store_id,
                        "item_id": item_id,
                        "wm_yr_wk": dup_week,
                        "sell_price": week_price_map[dup_week],
                    }
                )
                injected_duplicate_prices += 1

            daily_prices = np.array(
                [week_price_map.get(int(wk), base_price) for wk in wm_weeks],
                dtype=np.float64,
            )
            price_elasticity = (base_price / np.maximum(daily_prices, 0.1)) ** 1.4
            lift = np.ones(num_days, dtype=np.float64)
            lift = np.where(is_weekend, lift * 1.30, lift)
            lift = np.where(is_event, lift * 1.45, lift)
            if cat_id == "FOODS":
                lift = np.where(snap_vec == 1, lift * 1.20, lift)

            lam = item_popularity[item_idx] * s_scale * price_elasticity * lift
            daily_sales = rng.poisson(lam).astype(np.int64)
            if launch_day > 0:
                daily_sales[:launch_day] = 0

            if inject_dirty and item_idx in (0, 1):
                dirty_idx = min(num_days - 1, launch_day + 2 + item_idx)
                daily_sales[dirty_idx] = -3 if item_idx == 0 else -1
                injected_negative_cells += 1

            row: dict[str, Any] = {
                "id": f"{item_id}_{store_id}_validation",
                "item_id": item_id,
                "dept_id": dept_id,
                "cat_id": cat_id,
                "store_id": store_id,
                "state_id": state_id,
            }
            for col_name, val in zip(day_cols, daily_sales.tolist(), strict=True):
                row[col_name] = int(val)
            sales_rows.append(row)

    prices_df = pd.DataFrame(price_rows)
    sales_df = pd.DataFrame(sales_rows)

    calendar_file = raw_dir / "calendar.csv"
    prices_file = raw_dir / "sell_prices.csv"
    sales_file = raw_dir / "sales_train_validation.csv"

    cal_df.to_csv(calendar_file, index=False)
    prices_df.to_csv(prices_file, index=False)
    sales_df.to_csv(sales_file, index=False)

    manifest = {
        "source": "generated_m5_synthetic",
        "generated_at": utc_now_iso(),
        "seed": seed,
        "num_days": num_days,
        "items_per_store": items_per_store,
        "stores": [s for s, _ in STORE_SPEC],
        "inject_dirty": inject_dirty,
        "injected_negative_cells": injected_negative_cells,
        "injected_duplicate_prices": injected_duplicate_prices,
        "row_counts": {
            "calendar": int(len(cal_df)),
            "sell_prices": int(len(prices_df)),
            "sales_wide": int(len(sales_df)),
        },
    }
    (raw_dir / ".bootstrap_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    return {
        "source": "generated",
        "raw_dir": str(raw_dir),
        "calendar_file": str(calendar_file),
        "prices_file": str(prices_file),
        "sales_file": str(sales_file),
        "manifest": manifest,
    }
