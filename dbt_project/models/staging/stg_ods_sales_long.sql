SELECT
    id,
    item_id,
    dept_id,
    cat_id,
    store_id,
    state_id,
    d,
    CAST(sales AS INTEGER) AS raw_sales
FROM read_parquet(
    '{{ env_var("LAKEPULSE_LAKEHOUSE_DIR", "../data/lakehouse") }}/ods/ods_sales_long/**/*.parquet',
    hive_partitioning = true
)
