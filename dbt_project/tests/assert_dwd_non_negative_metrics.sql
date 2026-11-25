SELECT
    store_id,
    item_id,
    date,
    sales,
    sell_price,
    revenue
FROM {{ ref('dwd_daily_item_sales') }}
WHERE sales < 0
   OR sell_price <= 0
   OR revenue < 0
