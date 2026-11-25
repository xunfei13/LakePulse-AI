SELECT
    store_id,
    item_id,
    CAST(wm_yr_wk AS INTEGER) AS wm_yr_wk,
    CAST(sell_price AS DOUBLE) AS sell_price
FROM read_parquet(
    '{{ env_var("LAKEPULSE_LAKEHOUSE_DIR", "../data/lakehouse") }}/ods/ods_sell_prices/**/*.parquet'
)
