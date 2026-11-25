SELECT
    store_id,
    state_id,
    date,
    weekday,
    is_weekend,
    is_event_day,
    is_snap_day,
    COUNT(*) AS active_sku_count,
    SUM(sales) AS total_sales_qty,
    ROUND(SUM(revenue), 4) AS total_revenue,
    SUM(CASE WHEN sales = 0 THEN 1 ELSE 0 END) AS zero_sales_sku_count,
    SUM(is_repaired_negative) AS repaired_anomaly_count
FROM {{ ref('dwd_daily_item_sales') }}
GROUP BY
    store_id,
    state_id,
    date,
    weekday,
    is_weekend,
    is_event_day,
    is_snap_day
