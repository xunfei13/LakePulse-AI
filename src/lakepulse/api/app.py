"""FastAPI read-only data service gateway with OpenAPI tools for Dify Agent."""

from __future__ import annotations

from typing import Any

import duckdb
from fastapi import FastAPI, HTTPException, Query

from lakepulse.api.db import (
    SQLQueryTimeoutError,
    execute_readonly_query,
    list_whitelisted_tables,
)
from lakepulse.api.schemas import (
    AuditItem,
    ModelMetricItem,
    OpsHealthSummaryResponse,
    ReplenishmentAlertItem,
    ReplenishmentAlertsResponse,
    ServiceHealthResponse,
    SkuDemandAttributionResponse,
    SQLQueryRequest,
    SQLQueryResponse,
)
from lakepulse.api.sql_firewall import (
    SQLSecurityViolationError,
    SQLSyntaxValidationError,
    validate_and_rewrite_sql,
)
from lakepulse.config import AppSettings, get_settings


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """Create a configured FastAPI application instance."""
    cfg = settings or get_settings()

    app = FastAPI(
        title="LakePulse-AI Data & Replenishment Gateway",
        version="0.1.0",
        description=(
            "Read-only DuckDB lakehouse gateway guarded by a sqlglot AST firewall, "
            "providing pipeline health inspection, stockout alerts, SKU demand "
            "attribution, and controlled analytical SQL tools for Dify AI Agents."
        ),
    )

    @app.get(
        "/health",
        response_model=ServiceHealthResponse,
        operation_id="get_service_health",
        summary="Check API liveness and available whitelisted lakehouse tables",
    )
    def get_service_health() -> ServiceHealthResponse:
        tables = list_whitelisted_tables(cfg)
        return ServiceHealthResponse(
            status="ok",
            mode=cfg.mode,
            duckdb_ready=cfg.duckdb_path.is_file(),
            whitelisted_tables=tables,
        )

    @app.get(
        "/api/v1/ops/health-summary",
        response_model=OpsHealthSummaryResponse,
        operation_id="check_pipeline_and_model_health",
        summary="Inspect latest ETL/ML audit statuses and TEST-split model accuracy",
    )
    def check_pipeline_and_model_health() -> OpsHealthSummaryResponse:
        try:
            _, audit_rows = execute_readonly_query(
                cfg,
                """
                SELECT
                    run_id,
                    pipeline_stage,
                    table_name,
                    finished_at,
                    duration_sec,
                    output_rows,
                    null_rate,
                    status,
                    message
                FROM ops_pipeline_audit
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY pipeline_stage, table_name
                    ORDER BY finished_at DESC
                ) = 1
                ORDER BY finished_at DESC
                """,
            )
            _, metric_rows = execute_readonly_query(
                cfg,
                """
                SELECT
                    model_name,
                    eval_split,
                    sample_count,
                    mae,
                    rmse,
                    wape_pct,
                    r2_score,
                    is_best_test_wape,
                    deployed_for_inference
                FROM ops_model_eval_metrics
                WHERE eval_split = 'TEST'
                ORDER BY wape_pct ASC
                """,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except duckdb.Error as exc:
            raise HTTPException(status_code=503, detail=f"Lakehouse query error: {exc}") from exc

        failed_tables = [r["table_name"] for r in audit_rows if r["status"] == "FAILED"]
        warn_tables = [r["table_name"] for r in audit_rows if r["status"] == "WARN"]
        overall = "FAILED" if failed_tables else ("WARN" if warn_tables else "SUCCESS")

        return OpsHealthSummaryResponse(
            overall_status=overall,
            failed_tables=failed_tables,
            warn_tables=warn_tables,
            latest_audits=[AuditItem(**r) for r in audit_rows],
            test_model_metrics=[ModelMetricItem(**r) for r in metric_rows],
        )

    @app.get(
        "/api/v1/replenishment/alerts",
        response_model=ReplenishmentAlertsResponse,
        operation_id="get_replenishment_alerts",
        summary="Query prioritized stockout alerts and replenishment orders",
    )
    def get_replenishment_alerts(
        store_id: str | None = Query(default=None, description="Store ID e.g. CA_1"),
        abc_class: str | None = Query(default=None, description="ABC tier: A, B, or C"),
        risk_level: str | None = Query(
            default=None, description="Risk tier: HIGH, MEDIUM, or LOW"
        ),
        cat_id: str | None = Query(
            default=None, description="Category ID e.g. FOODS, HOBBIES, HOUSEHOLD"
        ),
        only_need_order: bool = Query(
            default=True, description="Filter to items with recommended_order_qty > 0"
        ),
        max_budget: float | None = Query(
            default=None,
            gt=0,
            description="Optional procurement budget cap ($) applied in priority order",
        ),
        limit: int = Query(default=50, ge=1, le=500, description="Max rows returned"),
    ) -> ReplenishmentAlertsResponse:
        clauses = ["1=1"]
        params: list[Any] = []

        if store_id:
            clauses.append("store_id = ?")
            params.append(store_id)
        if abc_class:
            clauses.append("abc_class = ?")
            params.append(abc_class.upper())
        if risk_level:
            clauses.append("stockout_risk_level = ?")
            params.append(risk_level.upper())
        if cat_id:
            clauses.append("cat_id = ?")
            params.append(cat_id.upper())
        if only_need_order:
            clauses.append("recommended_order_qty > 0")

        where_sql = " AND ".join(clauses)
        sql = f"""
            SELECT
                store_id,
                state_id,
                cat_id,
                dept_id,
                item_id,
                abc_class,
                forecast_start_date,
                sell_price,
                sales_ma_7,
                sales_ma_28,
                pred_next_7d_sales,
                safety_buffer_days,
                safety_stock_qty,
                current_available_stock,
                recommended_order_qty,
                estimated_order_amount,
                coverage_days,
                stockout_risk_level,
                key_driver_tags,
                model_version
            FROM ads_replenishment_plan
            WHERE {where_sql}
            ORDER BY
                CASE stockout_risk_level WHEN 'HIGH' THEN 1 WHEN 'MEDIUM' THEN 2 ELSE 3 END,
                CASE abc_class WHEN 'A' THEN 1 WHEN 'B' THEN 2 ELSE 3 END,
                recommended_order_qty DESC,
                item_id ASC
            LIMIT 5000
        """
        try:
            _, rows = execute_readonly_query(cfg, sql, params)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except duckdb.Error as exc:
            raise HTTPException(status_code=503, detail=f"Lakehouse query error: {exc}") from exc

        total_matched = len(rows)
        if max_budget is not None:
            selected: list[dict[str, Any]] = []
            running_spend = 0.0
            for row in rows:
                cost = float(row["estimated_order_amount"])
                if running_spend + cost <= max_budget + 1e-9:
                    selected.append(row)
                    running_spend += cost
            sliced = selected[:limit]
        else:
            sliced = rows[:limit]

        total_qty = sum(int(r["recommended_order_qty"]) for r in sliced)
        total_amount = round(sum(float(r["estimated_order_amount"]) for r in sliced), 2)

        return ReplenishmentAlertsResponse(
            total_matched=total_matched,
            returned_count=len(sliced),
            total_recommended_order_qty=total_qty,
            total_estimated_order_amount=total_amount,
            max_budget=max_budget,
            items=[ReplenishmentAlertItem(**r) for r in sliced],
        )

    @app.get(
        "/api/v1/replenishment/sku-attribution/{store_id}/{item_id}",
        response_model=SkuDemandAttributionResponse,
        operation_id="explain_sku_demand_drivers",
        summary="Explain demand drivers and safety-stock formula breakdown for a SKU",
    )
    def explain_sku_demand_drivers(
        store_id: str,
        item_id: str,
    ) -> SkuDemandAttributionResponse:
        sql = """
            SELECT
                p.store_id,
                p.state_id,
                p.cat_id,
                p.dept_id,
                p.item_id,
                p.abc_class,
                p.forecast_start_date,
                p.pred_next_7d_sales,
                p.safety_buffer_days,
                p.safety_stock_qty,
                p.current_available_stock,
                p.recommended_order_qty,
                p.estimated_order_amount,
                p.coverage_days,
                p.stockout_risk_level,
                p.key_driver_tags,
                p.sell_price,
                COALESCE(f.price_ma_28, p.sell_price) AS price_ma_28,
                COALESCE(f.price_discount_rate, 0.0) AS price_discount_rate,
                COALESCE(f.sales_lag_1, 0.0) AS sales_lag_1,
                COALESCE(f.sales_lag_7, 0.0) AS sales_lag_7,
                p.sales_ma_7,
                p.sales_ma_28,
                COALESCE(f.sales_std_28, 0.0) AS sales_std_28,
                COALESCE(f.sales_trend_ratio, 1.0) AS sales_trend_ratio,
                COALESCE(f.next_7d_weekend_days, 0) AS next_7d_weekend_days,
                COALESCE(f.next_7d_event_days, 0) AS next_7d_event_days,
                COALESCE(f.next_7d_snap_days, 0) AS next_7d_snap_days,
                COALESCE(c.weekend_lift_pct, 0.0) AS category_weekend_lift_pct,
                COALESCE(c.holiday_lift_pct, 0.0) AS category_holiday_lift_pct,
                COALESCE(c.snap_lift_pct, 0.0) AS category_snap_lift_pct,
                COALESCE(c.latest_month_mom_pct, 0.0) AS category_latest_mom_pct
            FROM ads_replenishment_plan AS p
            LEFT JOIN dws_item_rolling_features AS f
                ON p.store_id = f.store_id
               AND p.item_id = f.item_id
               AND f.sample_split = 'INFER'
            LEFT JOIN dws_calendar_effect_analysis AS c
                ON p.store_id = c.store_id
               AND p.cat_id = c.cat_id
            WHERE p.store_id = ?
              AND p.item_id = ?
            LIMIT 1
        """
        try:
            _, rows = execute_readonly_query(cfg, sql, [store_id, item_id])
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except duckdb.Error as exc:
            raise HTTPException(status_code=503, detail=f"Lakehouse query error: {exc}") from exc

        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"SKU '{store_id}/{item_id}' not found in ads_replenishment_plan.",
            )

        rec = rows[0]
        breakdown = (
            f"pred_7d({rec['pred_next_7d_sales']}) + "
            f"safety_stock({rec['safety_stock_qty']}, {rec['safety_buffer_days']}d) - "
            f"available_stock({rec['current_available_stock']}) -> "
            f"recommended_order_qty={rec['recommended_order_qty']}"
        )
        return SkuDemandAttributionResponse(**rec, replenishment_formula_breakdown=breakdown)

    @app.post(
        "/api/v1/sql/query",
        response_model=SQLQueryResponse,
        operation_id="execute_readonly_sql",
        summary="Execute a read-only SELECT query validated by sqlglot AST firewall",
    )
    def execute_readonly_sql(payload: SQLQueryRequest) -> SQLQueryResponse:
        effective_max_rows = min(payload.max_rows, cfg.sql_max_rows)
        try:
            rewritten_sql, referenced_tables = validate_and_rewrite_sql(
                payload.sql, max_rows=effective_max_rows
            )
        except SQLSyntaxValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SQLSecurityViolationError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

        try:
            columns, rows = execute_readonly_query(cfg, rewritten_sql)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except SQLQueryTimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except duckdb.Error as exc:
            raise HTTPException(status_code=400, detail=f"DuckDB execution error: {exc}") from exc

        return SQLQueryResponse(
            columns=columns,
            row_count=len(rows),
            rows=rows,
            referenced_tables=referenced_tables,
            executed_sql=rewritten_sql,
        )

    return app
