SELECT
    CAST(date AS DATE) AS cal_date,
    CAST(wm_yr_wk AS INTEGER) AS wm_yr_wk,
    weekday,
    CAST(wday AS INTEGER) AS wday,
    CAST(month AS INTEGER) AS month,
    CAST(year AS INTEGER) AS year,
    d,
    event_name_1,
    event_type_1,
    event_name_2,
    event_type_2,
    CAST(snap_CA AS INTEGER) AS snap_CA,
    CAST(snap_TX AS INTEGER) AS snap_TX,
    CAST(snap_WI AS INTEGER) AS snap_WI,
    CASE WHEN weekday IN ('Saturday', 'Sunday') THEN TRUE ELSE FALSE END AS is_weekend
FROM read_parquet(
    '{{ env_var("LAKEPULSE_LAKEHOUSE_DIR", "../data/lakehouse") }}/ods/ods_calendar/**/*.parquet'
)
