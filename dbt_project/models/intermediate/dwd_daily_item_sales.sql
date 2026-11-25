WITH sales AS (
    SELECT * FROM {{ ref('stg_ods_sales_long') }}
),
cal AS (
    SELECT * FROM {{ ref('stg_ods_calendar') }}
),
prices AS (
    SELECT * FROM {{ ref('stg_ods_sell_prices') }}
)

SELECT
    s.store_id,
    s.state_id,
    s.cat_id,
    s.dept_id,
    s.item_id,
    c.cal_date AS date,
    c.wm_yr_wk,
    c.weekday,
    c.wday,
    c.is_weekend,
    c.d,
    c.event_name_1,
    c.event_type_1,
    CASE WHEN c.event_name_1 IS NOT NULL THEN 1 ELSE 0 END AS is_event_day,
    CASE
        WHEN s.state_id = 'CA' THEN c.snap_CA
        WHEN s.state_id = 'TX' THEN c.snap_TX
        WHEN s.state_id = 'WI' THEN c.snap_WI
        ELSE 0
    END AS is_snap_day,
    p.sell_price,
    s.raw_sales,
    CASE WHEN s.raw_sales < 0 THEN 0 ELSE s.raw_sales END AS sales,
    CASE WHEN s.raw_sales < 0 THEN 1 ELSE 0 END AS is_repaired_negative,
    ROUND(
        (CASE WHEN s.raw_sales < 0 THEN 0 ELSE s.raw_sales END) * p.sell_price,
        4
    ) AS revenue
FROM sales AS s
INNER JOIN cal AS c
    ON s.d = c.d
INNER JOIN prices AS p
    ON s.store_id = p.store_id
   AND s.item_id = p.item_id
   AND c.wm_yr_wk = p.wm_yr_wk
WHERE p.sell_price IS NOT NULL
  AND p.sell_price > 0
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY s.store_id, s.item_id, c.cal_date
    ORDER BY s.id
) = 1
