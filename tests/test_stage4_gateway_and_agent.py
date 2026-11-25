"""Stage 4 tests for sqlglot AST firewall, FastAPI gateway, and Agent closed loop."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

from lakepulse.api.agent_workflow import (
    export_dify_openapi_schema,
    run_replenishment_agent_closed_loop,
)
from lakepulse.api.app import create_app
from lakepulse.api.sql_firewall import (
    SQLSecurityViolationError,
    SQLSyntaxValidationError,
    validate_and_rewrite_sql,
)
from lakepulse.config import AppSettings, get_settings, load_yaml_settings


def _build_mini_duckdb_settings(tmp_path: Path) -> AppSettings:
    """Create an isolated DuckDB fixture containing both whitelisted and restricted tables."""
    base_cfg = get_settings(mode="local")
    yaml_cfg = load_yaml_settings(base_cfg.project_root / "configs" / "settings.yaml")
    duckdb_path = tmp_path / "data" / "lakehouse" / "lakepulse.duckdb"
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(duckdb_path))
    try:
        con.execute(
            """
            CREATE TABLE ods_sales_long (store_id VARCHAR, item_id VARCHAR, sales INTEGER);
            INSERT INTO ods_sales_long VALUES ('CA_1', 'FOODS_1_001', 10);

            CREATE TABLE dwd_daily_item_sales (store_id VARCHAR, item_id VARCHAR, sales INTEGER);
            INSERT INTO dwd_daily_item_sales VALUES ('CA_1', 'FOODS_1_001', 10);

            CREATE TABLE ops_pipeline_audit (
                run_id VARCHAR, pipeline_stage VARCHAR, table_name VARCHAR,
                started_at VARCHAR, finished_at VARCHAR, duration_sec DOUBLE,
                input_rows BIGINT, output_rows BIGINT, null_rate DOUBLE,
                status VARCHAR, message VARCHAR
            );
            INSERT INTO ops_pipeline_audit VALUES
            ('r1', 'stage3_distributed_inference', 'ads_replenishment_plan',
             '2026-01-01T00:00:00Z', '2026-01-01T00:00:10Z', 10.0, 3, 3, 0.0,
             'SUCCESS', 'Scored 3 SKUs cleanly');

            CREATE TABLE ops_model_eval_metrics (
                run_id VARCHAR, evaluated_at VARCHAR, model_name VARCHAR,
                eval_split VARCHAR, sample_count BIGINT, mae DOUBLE, rmse DOUBLE,
                wape_pct DOUBLE, r2_score DOUBLE,
                is_best_test_wape BOOLEAN, deployed_for_inference BOOLEAN
            );
            INSERT INTO ops_model_eval_metrics VALUES
            ('r1', '2026-01-01T00:00:05Z', 'tensorflow_wide_and_deep',
             'TEST', 100, 3.1, 4.2, 22.5, 0.86, TRUE, TRUE);

            CREATE TABLE dws_calendar_effect_analysis (
                store_id VARCHAR, state_id VARCHAR, cat_id VARCHAR,
                weekday_avg_sales DOUBLE, weekend_avg_sales DOUBLE,
                non_event_avg_sales DOUBLE, event_day_avg_sales DOUBLE,
                non_snap_avg_sales DOUBLE, snap_day_avg_sales DOUBLE,
                weekend_lift_pct DOUBLE, holiday_lift_pct DOUBLE,
                snap_lift_pct DOUBLE, latest_month_mom_pct DOUBLE
            );
            INSERT INTO dws_calendar_effect_analysis VALUES
            ('CA_1', 'CA', 'FOODS', 3.0, 4.0, 3.1, 4.6, 3.0, 3.7, 33.3, 48.4, 23.3, 1.5);

            CREATE TABLE dws_item_rolling_features (
                store_id VARCHAR, item_id VARCHAR, sample_split VARCHAR,
                price_ma_28 DOUBLE, price_discount_rate DOUBLE,
                sales_lag_1 DOUBLE, sales_lag_7 DOUBLE, sales_std_28 DOUBLE,
                sales_trend_ratio DOUBLE,
                next_7d_weekend_days INTEGER, next_7d_event_days INTEGER,
                next_7d_snap_days INTEGER
            );
            INSERT INTO dws_item_rolling_features VALUES
            ('CA_1', 'FOODS_1_001', 'INFER', 5.0, 0.10, 4.0, 3.5, 1.2, 1.25, 2, 1, 7),
            ('CA_1', 'FOODS_1_002', 'INFER', 8.0, 0.00, 3.0, 3.0, 0.9, 1.05, 2, 1, 7);

            CREATE TABLE ads_replenishment_plan (
                store_id VARCHAR, state_id VARCHAR, cat_id VARCHAR, dept_id VARCHAR,
                item_id VARCHAR, abc_class VARCHAR, forecast_start_date DATE,
                sell_price DOUBLE, sales_ma_7 DOUBLE, sales_ma_28 DOUBLE,
                pred_next_7d_sales DOUBLE, safety_buffer_days DOUBLE,
                safety_stock_qty DOUBLE, current_available_stock INTEGER,
                recommended_order_qty INTEGER, estimated_order_amount DOUBLE,
                coverage_days DOUBLE, stockout_risk_level VARCHAR,
                key_driver_tags VARCHAR, model_version VARCHAR
            );
            INSERT INTO ads_replenishment_plan VALUES
            ('CA_1', 'CA', 'FOODS', 'FOODS_1', 'FOODS_1_001', 'A', DATE '2016-01-01',
             4.5, 3.5, 2.8, 28.0, 3.0, 12.0, 4, 36, 162.0, 1.0, 'HIGH',
             'CORE_A_SKU,PRICE_PROMO,UPCOMING_HOLIDAY', 'tensorflow_wide_and_deep'),
            ('CA_1', 'CA', 'FOODS', 'FOODS_1', 'FOODS_1_002', 'A', DATE '2016-01-01',
             8.0, 3.0, 2.9, 21.0, 3.0, 9.0, 6, 24, 192.0, 2.0, 'HIGH',
             'CORE_A_SKU,UPCOMING_HOLIDAY', 'tensorflow_wide_and_deep'),
            ('CA_1', 'CA', 'FOODS', 'FOODS_1', 'FOODS_1_003', 'B', DATE '2016-01-01',
             3.0, 1.0, 1.0, 7.0, 2.0, 2.0, 12, 0, 0.0, 12.0, 'LOW',
             'REGULAR_DEMAND', 'tensorflow_wide_and_deep');
            """
        )
    finally:
        con.close()

    return AppSettings(
        mode="local",
        project_root=base_cfg.project_root,
        data_dir_raw=tmp_path / "data",
        duckdb_path_raw=duckdb_path,
        models_dir_raw=tmp_path / "artifacts" / "models",
        sql_max_rows=500,
        yaml_config=yaml_cfg,
    )


def test_sql_firewall_allows_whitelisted_select_cte_and_clamps_limit() -> None:
    """Verify SELECT, CTE, UNION ALL pass firewall and row limits are strictly capped."""
    sql_no_limit = "SELECT store_id, item_id FROM ads_replenishment_plan"
    rewritten, tables = validate_and_rewrite_sql(sql_no_limit, max_rows=200)
    assert rewritten.endswith("LIMIT 200")
    assert tables == ["ads_replenishment_plan"]

    sql_cte = (
        "WITH high_risk AS ("
        "  SELECT item_id, recommended_order_qty FROM ads_replenishment_plan"
        ") SELECT * FROM high_risk LIMIT 99999"
    )
    rewritten_cte, tables_cte = validate_and_rewrite_sql(sql_cte, max_rows=500)
    assert rewritten_cte.endswith("LIMIT 500")
    assert tables_cte == ["ads_replenishment_plan"]

    sql_small_limit = "SELECT item_id FROM dws_item_abc_profile LIMIT 5"
    rewritten_small, _ = validate_and_rewrite_sql(sql_small_limit, max_rows=500)
    assert rewritten_small.endswith("LIMIT 5")

    sql_union = (
        "SELECT table_name FROM ops_pipeline_audit "
        "UNION ALL SELECT model_name FROM ops_model_eval_metrics"
    )
    _, union_tables = validate_and_rewrite_sql(sql_union, max_rows=100)
    assert union_tables == ["ops_model_eval_metrics", "ops_pipeline_audit"]


@pytest.mark.parametrize(
    "bad_sql",
    [
        "DROP TABLE ads_replenishment_plan",
        "DELETE FROM ads_replenishment_plan WHERE store_id = 'CA_1'",
        "UPDATE ads_replenishment_plan SET recommended_order_qty = 0",
        "INSERT INTO ads_replenishment_plan VALUES ('CA_1')",
        "ALTER TABLE ads_replenishment_plan ADD COLUMN x INT",
        "CREATE TABLE evil (id INT)",
        "COPY ads_replenishment_plan TO 'stolen.csv'",
        "ATTACH 'other.db' AS other",
        "PRAGMA show_tables",
        "SELECT * FROM ads_replenishment_plan; DROP TABLE ops_pipeline_audit",
        "SELECT * FROM read_parquet('secret.parquet')",
        "SELECT * FROM read_csv_auto('secret.csv')",
        "SELECT * FROM duckdb_tables()",
    ],
)
def test_sql_firewall_blocks_ddl_dml_multistatement_and_file_readers(bad_sql: str) -> None:
    """Verify dangerous SQL operations, injections, and file readers raise security error."""
    with pytest.raises(SQLSecurityViolationError):
        validate_and_rewrite_sql(bad_sql, max_rows=500)


@pytest.mark.parametrize(
    "restricted_sql",
    [
        "SELECT * FROM ods_sales_long LIMIT 5",
        "SELECT * FROM dwd_daily_item_sales LIMIT 5",
        "SELECT * FROM stg_ods_calendar LIMIT 5",
        "SELECT * FROM information_schema.tables",
    ],
)
def test_sql_firewall_blocks_non_whitelisted_layers(restricted_sql: str) -> None:
    """Verify non-whitelisted layers and system catalogs are rejected."""
    with pytest.raises(SQLSecurityViolationError):
        validate_and_rewrite_sql(restricted_sql, max_rows=500)


def test_sql_firewall_rejects_empty_and_syntax_errors() -> None:
    """Verify empty strings and invalid syntax raise SQLSyntaxValidationError."""
    with pytest.raises(SQLSyntaxValidationError):
        validate_and_rewrite_sql("   ")
    with pytest.raises(SQLSyntaxValidationError):
        validate_and_rewrite_sql("SELECT FROM WHERE ???")


def test_fastapi_gateway_endpoints_and_agent_closed_loop(tmp_path: Path) -> None:
    """Verify FastAPI endpoints, HTTP 400/403/404 handling, OpenAPI export, and Agent loop."""
    cfg = _build_mini_duckdb_settings(tmp_path)
    app = create_app(cfg)

    openapi_out = tmp_path / "openapi_tools.json"
    export_dify_openapi_schema(openapi_out, settings=cfg)
    assert openapi_out.is_file() and openapi_out.stat().st_size > 100

    with TestClient(app) as client:
        r_health = client.get("/health")
        assert r_health.status_code == 200
        assert "ads_replenishment_plan" in r_health.json()["whitelisted_tables"]
        assert "dwd_daily_item_sales" not in r_health.json()["whitelisted_tables"]

        r_ops = client.get("/api/v1/ops/health-summary")
        assert r_ops.status_code == 200
        assert r_ops.json()["overall_status"] == "SUCCESS"

        # Budget cap 200.0 fits FOODS_1_001 ($162.0) but excludes FOODS_1_002 ($192.0)
        r_alerts = client.get(
            "/api/v1/replenishment/alerts",
            params={"store_id": "CA_1", "abc_class": "A", "max_budget": 200.0},
        )
        assert r_alerts.status_code == 200
        alerts_json = r_alerts.json()
        assert alerts_json["total_matched"] == 2
        assert alerts_json["returned_count"] == 1
        assert alerts_json["total_estimated_order_amount"] <= 200.0

        r_attr = client.get("/api/v1/replenishment/sku-attribution/CA_1/FOODS_1_001")
        assert r_attr.status_code == 200
        assert "recommended_order_qty=36" in r_attr.json()["replenishment_formula_breakdown"]

        r_missing = client.get("/api/v1/replenishment/sku-attribution/CA_1/NO_SUCH_SKU")
        assert r_missing.status_code == 404

        r_sql_ok = client.post(
            "/api/v1/sql/query",
            json={"sql": "SELECT item_id, recommended_order_qty FROM ads_replenishment_plan"},
        )
        assert r_sql_ok.status_code == 200
        assert r_sql_ok.json()["row_count"] == 3

        r_sql_forbidden = client.post(
            "/api/v1/sql/query",
            json={"sql": "SELECT * FROM dwd_daily_item_sales"},
        )
        assert r_sql_forbidden.status_code == 403

        r_sql_syntax = client.post(
            "/api/v1/sql/query",
            json={"sql": "SELECT FROM ???"},
        )
        assert r_sql_syntax.status_code == 400

    agent_res = run_replenishment_agent_closed_loop(
        cfg,
        store_id="CA_1",
        abc_class="A",
        max_budget=400.0,
    )
    assert agent_res["alerts"]["returned_count"] == 2
    assert agent_res["alerts"]["total_estimated_order_amount"] <= 400.0
    assert len(agent_res["attributions"]) == 2
    assert "智能补货审批工单" in agent_res["workorder_markdown"]
