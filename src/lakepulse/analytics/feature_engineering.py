"""Leak-free rolling window feature engineering for retail demand forecasting."""

from __future__ import annotations

from datetime import date, timedelta

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, IntegerType, StructField, StructType

from lakepulse.config import AppSettings
from lakepulse.etl.m5_bootstrap import HOLIDAY_EVENTS

FEATURE_COLUMNS: tuple[str, ...] = (
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_ma_7",
    "sales_ma_28",
    "sales_std_28",
    "sales_trend_ratio",
    "sell_price",
    "price_ma_28",
    "price_discount_rate",
    "is_weekend",
    "is_event_day",
    "is_snap_day",
    "next_7d_weekend_days",
    "next_7d_event_days",
    "next_7d_snap_days",
)

FUTURE_CAL_BASE_SCHEMA = StructType(
    [
        StructField("cal_date", DateType(), nullable=False),
        StructField("is_weekend_int", IntegerType(), nullable=False),
        StructField("is_event_day", IntegerType(), nullable=False),
        StructField("snap_CA", IntegerType(), nullable=False),
        StructField("snap_TX", IntegerType(), nullable=False),
        StructField("snap_WI", IntegerType(), nullable=False),
    ]
)


def build_future_calendar_schedule(
    spark: SparkSession,
    df_cal: DataFrame,
    max_sales_date: date,
    horizon: int = 7,
) -> DataFrame:
    """Compute known [T+1, T+horizon] calendar schedule counts for every date T."""
    df_base = df_cal.select(
        F.to_date(F.col("date")).alias("cal_date"),
        F.when(F.col("weekday").isin("Saturday", "Sunday"), F.lit(1))
        .otherwise(F.lit(0))
        .cast("int")
        .alias("is_weekend_int"),
        F.when(
            F.col("event_name_1").isNotNull() & (F.trim(F.col("event_name_1")) != ""),
            F.lit(1),
        )
        .otherwise(F.lit(0))
        .cast("int")
        .alias("is_event_day"),
        F.coalesce(F.col("snap_CA").cast("int"), F.lit(0)).alias("snap_CA"),
        F.coalesce(F.col("snap_TX").cast("int"), F.lit(0)).alias("snap_TX"),
        F.coalesce(F.col("snap_WI").cast("int"), F.lit(0)).alias("snap_WI"),
    ).dropDuplicates(["cal_date"])

    cal_max_date = df_base.agg(F.max("cal_date")).collect()[0][0]
    required_end = max_sales_date + timedelta(days=horizon)

    if cal_max_date is not None and cal_max_date < required_end:
        ext_rows: list[tuple[date, int, int, int, int, int]] = []
        cur = cal_max_date + timedelta(days=1)
        while cur <= required_end:
            is_wknd = 1 if cur.weekday() >= 5 else 0
            is_evt = 1 if (cur.month, cur.day) in HOLIDAY_EVENTS else 0
            snap = 1 if cur.day <= 10 else 0
            ext_rows.append((cur, is_wknd, is_evt, snap, snap, snap))
            cur += timedelta(days=1)
        df_ext = spark.createDataFrame(ext_rows, schema=FUTURE_CAL_BASE_SCHEMA)
        df_base = df_base.unionByName(df_ext)

    w_fwd = Window.orderBy("cal_date").rowsBetween(1, horizon)
    return df_base.select(
        F.col("cal_date"),
        F.coalesce(F.sum("is_weekend_int").over(w_fwd), F.lit(0))
        .cast("int")
        .alias("next_7d_weekend_days"),
        F.coalesce(F.sum("is_event_day").over(w_fwd), F.lit(0))
        .cast("int")
        .alias("next_7d_event_days"),
        F.coalesce(F.sum("snap_CA").over(w_fwd), F.lit(0))
        .cast("int")
        .alias("next_7d_snap_CA"),
        F.coalesce(F.sum("snap_TX").over(w_fwd), F.lit(0))
        .cast("int")
        .alias("next_7d_snap_TX"),
        F.coalesce(F.sum("snap_WI").over(w_fwd), F.lit(0))
        .cast("int")
        .alias("next_7d_snap_WI"),
    )


def build_rolling_features_df(
    spark: SparkSession,
    df_dwd: DataFrame,
    df_cal: DataFrame,
    settings: AppSettings,
) -> DataFrame:
    """Build leak-free tabular rolling features and embargoed time splits."""
    win_cfg = settings.yaml_config.get("analytics", {}).get("windows", {})
    short_window = int(win_cfg.get("short_window", 7))
    long_window = int(win_cfg.get("long_window", 28))
    horizon = int(win_cfg.get("forecast_horizon", 7))

    df = (
        df_dwd.withColumn("date", F.to_date(F.col("date")))
        .withColumn("is_weekend", F.col("is_weekend").cast("int"))
        .withColumn("is_event_day", F.col("is_event_day").cast("int"))
        .withColumn("is_snap_day", F.col("is_snap_day").cast("int"))
        .withColumn("month", F.month(F.col("date")).cast("int"))
    )

    bounds = df.agg(F.min("date").alias("min_d"), F.max("date").alias("max_d")).collect()[0]
    min_sales_date: date = bounds["min_d"]
    max_sales_date: date = bounds["max_d"]

    df_future_cal = build_future_calendar_schedule(
        spark,
        df_cal,
        max_sales_date=max_sales_date,
        horizon=horizon,
    )

    w_item = Window.partitionBy("store_id", "item_id").orderBy("date")
    w_short = w_item.rowsBetween(-(short_window - 1), 0)
    w_long = w_item.rowsBetween(-(long_window - 1), 0)
    w_fwd = w_item.rowsBetween(1, horizon)

    df_feat = (
        df.withColumn("item_obs_days", F.row_number().over(w_item))
        .withColumn("sales_lag_1", F.col("sales").cast("double"))
        .withColumn("sales_ma_7", F.round(F.avg("sales").over(w_short), 4))
        .withColumn("sales_ma_28", F.round(F.avg("sales").over(w_long), 4))
        .withColumn(
            "sales_lag_7",
            F.coalesce(F.lag("sales", 7).over(w_item).cast("double"), F.col("sales_ma_7")),
        )
        .withColumn(
            "sales_lag_14",
            F.coalesce(F.lag("sales", 14).over(w_item).cast("double"), F.col("sales_ma_28")),
        )
        .withColumn(
            "sales_std_28",
            F.round(F.coalesce(F.stddev_samp("sales").over(w_long), F.lit(0.0)), 4),
        )
        .withColumn(
            "sales_trend_ratio",
            F.round(
                F.col("sales_ma_7") / F.greatest(F.col("sales_ma_28"), F.lit(0.01)),
                4,
            ),
        )
        .withColumn("price_ma_28", F.round(F.avg("sell_price").over(w_long), 4))
        .withColumn(
            "price_discount_rate",
            F.round(
                (F.col("price_ma_28") - F.col("sell_price"))
                / F.greatest(F.col("price_ma_28"), F.lit(0.01)),
                4,
            ),
        )
        .withColumn(
            "target_next_7d_sales",
            F.when(
                (F.count("sales").over(w_fwd) == F.lit(horizon))
                & (F.datediff(F.max("date").over(w_fwd), F.col("date")) == F.lit(horizon)),
                F.round(F.sum("sales").over(w_fwd).cast("double"), 4),
            ).otherwise(F.lit(None).cast("double")),
        )
    )

    df_joined = (
        df_feat.join(
            F.broadcast(df_future_cal),
            df_feat["date"] == df_future_cal["cal_date"],
            how="left",
        )
        .drop("cal_date")
        .withColumn(
            "next_7d_snap_days",
            F.when(F.col("state_id") == "CA", F.col("next_7d_snap_CA"))
            .when(F.col("state_id") == "TX", F.col("next_7d_snap_TX"))
            .when(F.col("state_id") == "WI", F.col("next_7d_snap_WI"))
            .otherwise(F.lit(0)),
        )
        .fillna(
            {
                "next_7d_weekend_days": 0,
                "next_7d_event_days": 0,
                "next_7d_snap_days": 0,
            }
        )
        .drop("next_7d_snap_CA", "next_7d_snap_TX", "next_7d_snap_WI")
    )

    df_ready = df_joined.filter(
        (F.col("item_obs_days") >= F.lit(long_window)) | (F.col("date") == F.lit(max_sales_date))
    )

    span_days = (max_sales_date - min_sales_date).days + 1
    eval_len = 14 if span_days >= 120 else max(3, span_days // 10)
    test_end = max_sales_date - timedelta(days=horizon)
    test_start = test_end - timedelta(days=eval_len - 1)
    val_end = test_start - timedelta(days=horizon + 1)
    val_start = val_end - timedelta(days=eval_len - 1)
    train_end = val_start - timedelta(days=horizon + 1)

    df_split = df_ready.withColumn(
        "sample_split",
        F.when(F.col("date") == F.lit(max_sales_date), F.lit("INFER"))
        .when(
            (F.col("date") >= F.lit(test_start))
            & (F.col("date") <= F.lit(test_end))
            & F.col("target_next_7d_sales").isNotNull(),
            F.lit("TEST"),
        )
        .when(
            (F.col("date") >= F.lit(val_start))
            & (F.col("date") <= F.lit(val_end))
            & F.col("target_next_7d_sales").isNotNull(),
            F.lit("VAL"),
        )
        .when(
            (F.col("date") <= F.lit(train_end))
            & F.col("target_next_7d_sales").isNotNull(),
            F.lit("TRAIN"),
        )
        .otherwise(F.lit("EMBARGO")),
    )

    return df_split.select(
        "store_id",
        "state_id",
        "cat_id",
        "dept_id",
        "item_id",
        "date",
        "wm_yr_wk",
        "wday",
        "month",
        "sample_split",
        "sales",
        *FEATURE_COLUMNS,
        "target_next_7d_sales",
    )
