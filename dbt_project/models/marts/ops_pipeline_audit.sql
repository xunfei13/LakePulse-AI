SELECT
    run_id,
    pipeline_stage,
    table_name,
    started_at,
    finished_at,
    duration_sec,
    input_rows,
    output_rows,
    null_rate,
    status,
    message
FROM read_parquet(
    '{{ env_var("LAKEPULSE_LAKEHOUSE_DIR", "../data/lakehouse") }}/ops/ops_pipeline_audit/**/*.parquet'
)
