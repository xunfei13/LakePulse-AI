SELECT
    store_id,
    item_id,
    date,
    COUNT(*) AS row_cnt
FROM {{ ref('dwd_daily_item_sales') }}
GROUP BY store_id, item_id, date
HAVING COUNT(*) > 1
