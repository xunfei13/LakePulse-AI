"""Retail commercial diagnostics: ABC segmentation, calendar lift, and price elasticity."""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from lakepulse.config import AppSettings


def build_abc_profile(df_dwd: DataFrame, settings: AppSettings) -> DataFrame:
    """Build store-level ABC item value segmentation and safety buffer day mapping."""
    analytics_cfg = settings.yaml_config.get("analytics", {})
    abc_thresh = analytics_cfg.get("abc_thresholds", {})
    a_thresh = float(abc_thresh.get("A", 0.70))
    b_thresh = float(abc_thresh.get("B", 0.90))

    repl_cfg = settings.yaml_config.get("replenishment", {})
    buf_days = repl_cfg.get("safety_buffer_days", {})
    buf_a = float(buf_days.get("A", 3.0))
    buf_b = float(buf_days.get("B", 2.0))
    buf_c = float(buf_days.get("C", 1.0))

    df_item = df_dwd.groupBy("store_id", "state_id", "cat_id", "dept_id", "item_id").agg(
        F.count("date").cast("long").alias("active_days"),
        F.sum("sales").cast("long").alias("total_sales_qty"),
        F.round(F.sum("revenue"), 4).alias("total_revenue"),
        F.round(F.avg("sales"), 4).alias("avg_daily_sales"),
        F.round(F.avg("sell_price"), 4).alias("avg_sell_price"),
    )

    w_store = Window.partitionBy("store_id")
    w_rank = Window.partitionBy("store_id").orderBy(
        F.col("total_revenue").desc(), F.col("item_id").asc()
    )
    w_cum = w_rank.rowsBetween(Window.unboundedPreceding, 0)

    df_ranked = (
        df_item.withColumn("store_total_revenue", F.sum("total_revenue").over(w_store))
        .withColumn("revenue_rank", F.row_number().over(w_rank))
        .withColumn(
            "store_revenue_share",
            F.round(
                F.col("total_revenue") / F.greatest(F.col("store_total_revenue"), F.lit(0.01)),
                6,
            ),
        )
        .withColumn(
            "cum_revenue_share",
            F.round(
                F.sum("total_revenue").over(w_cum)
                / F.greatest(F.col("store_total_revenue"), F.lit(0.01)),
                6,
            ),
        )
    )

    df_abc = df_ranked.withColumn(
        "abc_class",
        F.when(
            (F.col("cum_revenue_share") <= F.lit(a_thresh)) | (F.col("revenue_rank") == 1),
            F.lit("A"),
        )
        .when(F.col("cum_revenue_share") <= F.lit(b_thresh), F.lit("B"))
        .otherwise(F.lit("C")),
    ).withColumn(
        "safety_buffer_days",
        F.when(F.col("abc_class") == "A", F.lit(buf_a))
        .when(F.col("abc_class") == "B", F.lit(buf_b))
        .otherwise(F.lit(buf_c)),
    )

    return df_abc.drop("store_total_revenue")


def build_calendar_effect_analysis(df_dwd: DataFrame) -> DataFrame:
    """Quantify weekend, holiday event, SNAP coupon, and MoM revenue dynamics."""
    df = df_dwd.withColumn("date", F.to_date(F.col("date"))).withColumn(
        "is_weekend_int", F.col("is_weekend").cast("int")
    )

    df_stats = df.groupBy("store_id", "state_id", "cat_id").agg(
        F.round(
            F.coalesce(F.avg(F.when(F.col("is_weekend_int") == 0, F.col("sales"))), F.lit(0.0)),
            4,
        ).alias("weekday_avg_sales"),
        F.round(
            F.coalesce(F.avg(F.when(F.col("is_weekend_int") == 1, F.col("sales"))), F.lit(0.0)),
            4,
        ).alias("weekend_avg_sales"),
        F.round(
            F.coalesce(F.avg(F.when(F.col("is_event_day") == 0, F.col("sales"))), F.lit(0.0)),
            4,
        ).alias("non_event_avg_sales"),
        F.round(
            F.coalesce(F.avg(F.when(F.col("is_event_day") == 1, F.col("sales"))), F.lit(0.0)),
            4,
        ).alias("event_day_avg_sales"),
        F.sum(F.when(F.col("is_event_day") == 1, 1).otherwise(0)).alias("_event_obs"),
        F.round(
            F.coalesce(F.avg(F.when(F.col("is_snap_day") == 0, F.col("sales"))), F.lit(0.0)),
            4,
        ).alias("non_snap_avg_sales"),
        F.round(
            F.coalesce(F.avg(F.when(F.col("is_snap_day") == 1, F.col("sales"))), F.lit(0.0)),
            4,
        ).alias("snap_day_avg_sales"),
        F.sum(F.when(F.col("is_snap_day") == 1, 1).otherwise(0)).alias("_snap_obs"),
    )

    df_lifts = (
        df_stats.withColumn(
            "weekend_lift_pct",
            F.round(
                (F.col("weekend_avg_sales") - F.col("weekday_avg_sales"))
                / F.greatest(F.col("weekday_avg_sales"), F.lit(0.01))
                * 100.0,
                2,
            ),
        )
        .withColumn(
            "holiday_lift_pct",
            F.when(
                F.col("_event_obs") > 0,
                F.round(
                    (F.col("event_day_avg_sales") - F.col("non_event_avg_sales"))
                    / F.greatest(F.col("non_event_avg_sales"), F.lit(0.01))
                    * 100.0,
                    2,
                ),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "snap_lift_pct",
            F.when(
                F.col("_snap_obs") > 0,
                F.round(
                    (F.col("snap_day_avg_sales") - F.col("non_snap_avg_sales"))
                    / F.greatest(F.col("non_snap_avg_sales"), F.lit(0.01))
                    * 100.0,
                    2,
                ),
            ).otherwise(F.lit(0.0)),
        )
        .drop("_event_obs", "_snap_obs")
    )

    df_monthly = df.withColumn("month_start", F.date_trunc("month", F.col("date"))).groupBy(
        "store_id", "state_id", "cat_id", "month_start"
    ).agg(
        (F.sum("revenue") / F.greatest(F.countDistinct("date"), F.lit(1))).alias(
            "monthly_daily_avg_rev"
        )
    )
    w_month = Window.partitionBy("store_id", "state_id", "cat_id").orderBy(
        F.col("month_start").desc()
    )
    df_mom = (
        df_monthly.withColumn("rn", F.row_number().over(w_month))
        .withColumn("prev_month_rev", F.lead("monthly_daily_avg_rev", 1).over(w_month))
        .filter(F.col("rn") == 1)
        .withColumn(
            "latest_month_mom_pct",
            F.when(
                F.col("prev_month_rev").isNotNull() & (F.col("prev_month_rev") > 0),
                F.round(
                    (F.col("monthly_daily_avg_rev") - F.col("prev_month_rev"))
                    / F.col("prev_month_rev")
                    * 100.0,
                    2,
                ),
            ).otherwise(F.lit(0.0)),
        )
        .select("store_id", "state_id", "cat_id", "latest_month_mom_pct")
    )

    return df_lifts.join(df_mom, on=["store_id", "state_id", "cat_id"], how="left").fillna(
        {"latest_month_mom_pct": 0.0}
    )


def build_price_elasticity_analysis(
    df_dwd: DataFrame,
    long_window: int = 28,
) -> DataFrame:
    """Measure sales response across discount depth tiers using trailing 28-day price mean."""
    df = df_dwd.withColumn("date", F.to_date(F.col("date")))
    w_item = Window.partitionBy("store_id", "item_id").orderBy("date")
    w_28 = w_item.rowsBetween(-(long_window - 1), 0)
    w_item_all = Window.partitionBy("store_id", "item_id")

    df_priced = (
        df.withColumn("price_ma_28", F.avg("sell_price").over(w_28))
        .withColumn(
            "price_discount_rate",
            (F.col("price_ma_28") - F.col("sell_price"))
            / F.greatest(F.col("price_ma_28"), F.lit(0.01)),
        )
        .withColumn(
            "item_regular_avg_sales",
            F.coalesce(
                F.avg(
                    F.when(F.col("price_discount_rate") <= 0.01, F.col("sales"))
                ).over(w_item_all),
                F.avg("sales").over(w_item_all),
            ),
        )
        .withColumn(
            "discount_bucket",
            F.when(F.col("price_discount_rate") <= 0.01, F.lit("0_NO_DISCOUNT"))
            .when(F.col("price_discount_rate") <= 0.10, F.lit("1_MILD_PROMO_1_10PCT"))
            .otherwise(F.lit("2_DEEP_PROMO_GT_10PCT")),
        )
    )

    return df_priced.groupBy("store_id", "cat_id", "discount_bucket").agg(
        F.count("*").cast("long").alias("obs_item_days"),
        F.round(F.avg("price_discount_rate") * 100.0, 2).alias("avg_discount_pct"),
        F.round(F.avg("sales"), 4).alias("avg_daily_sales"),
        F.round(F.avg("revenue"), 4).alias("avg_daily_revenue"),
        F.round(
            F.avg(
                (F.col("sales") - F.col("item_regular_avg_sales"))
                / F.greatest(F.col("item_regular_avg_sales"), F.lit(0.1))
            )
            * 100.0,
            2,
        ).alias("avg_item_sales_lift_pct"),
    )
