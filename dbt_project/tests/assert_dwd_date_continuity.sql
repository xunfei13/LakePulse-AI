SELECT
    store_id,
    MIN(date) AS min_date,
    MAX(date) AS max_date,
    COUNT(DISTINCT date) AS distinct_days,
    DATEDIFF('day', MIN(date), MAX(date)) + 1 AS expected_days
FROM {{ ref('dws_store_daily_summary') }}
GROUP BY store_id
HAVING COUNT(DISTINCT date) <> (DATEDIFF('day', MIN(date), MAX(date)) + 1)
