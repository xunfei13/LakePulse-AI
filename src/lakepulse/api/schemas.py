"""Pydantic schemas defining FastAPI & OpenAPI contracts for LakePulse-AI."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ServiceHealthResponse(BaseModel):
    """Service liveness and available whitelisted lakehouse tables."""

    status: str = Field(description="Service status ('ok').")
    mode: str = Field(description="Configured runtime mode ('local' or 'cluster').")
    duckdb_ready: bool = Field(description="Whether DuckDB lakehouse file exists.")
    whitelisted_tables: list[str] = Field(
        description="Accessible ads_*, dws_*, and ops_* tables."
    )


class AuditItem(BaseModel):
    """Latest pipeline execution audit summary for a single table."""

    run_id: str
    pipeline_stage: str
    table_name: str
    finished_at: str
    duration_sec: float
    output_rows: int
    null_rate: float
    status: str
    message: str


class ModelMetricItem(BaseModel):
    """Model evaluation metrics on the held-out TEST split."""

    model_name: str
    eval_split: str
    sample_count: int
    mae: float
    rmse: float
    wape_pct: float
    r2_score: float
    is_best_test_wape: bool
    deployed_for_inference: bool


class OpsHealthSummaryResponse(BaseModel):
    """Aggregated pipeline SLA audit and model performance summary."""

    overall_status: str = Field(description="Overall health: SUCCESS, WARN, or FAILED.")
    failed_tables: list[str] = Field(description="Tables whose latest run ended in FAILED.")
    warn_tables: list[str] = Field(description="Tables whose latest run emitted WARN.")
    latest_audits: list[AuditItem]
    test_model_metrics: list[ModelMetricItem]


class ReplenishmentAlertItem(BaseModel):
    """Single store-SKU replenishment recommendation record."""

    store_id: str
    state_id: str
    cat_id: str
    dept_id: str
    item_id: str
    abc_class: str
    forecast_start_date: str
    sell_price: float
    sales_ma_7: float
    sales_ma_28: float
    pred_next_7d_sales: float
    safety_buffer_days: float
    safety_stock_qty: float
    current_available_stock: int
    recommended_order_qty: int
    estimated_order_amount: float
    coverage_days: float
    stockout_risk_level: str
    key_driver_tags: str
    model_version: str


class ReplenishmentAlertsResponse(BaseModel):
    """Filtered and optionally budget-constrained replenishment priority list."""

    total_matched: int = Field(description="Total matching SKUs before budget/limit slice.")
    returned_count: int = Field(description="Number of SKUs returned after filtering.")
    total_recommended_order_qty: int = Field(description="Sum of recommended order units.")
    total_estimated_order_amount: float = Field(
        description="Total estimated procurement cost ($)."
    )
    max_budget: float | None = Field(
        default=None, description="Optional procurement budget cap applied."
    )
    items: list[ReplenishmentAlertItem]


class SkuDemandAttributionResponse(BaseModel):
    """Detailed demand drivers and safety-stock formula breakdown for one SKU."""

    store_id: str
    state_id: str
    cat_id: str
    dept_id: str
    item_id: str
    abc_class: str
    forecast_start_date: str
    pred_next_7d_sales: float
    safety_buffer_days: float
    safety_stock_qty: float
    current_available_stock: int
    recommended_order_qty: int
    estimated_order_amount: float
    coverage_days: float
    stockout_risk_level: str
    key_driver_tags: str
    sell_price: float
    price_ma_28: float
    price_discount_rate: float
    sales_lag_1: float
    sales_lag_7: float
    sales_ma_7: float
    sales_ma_28: float
    sales_std_28: float
    sales_trend_ratio: float
    next_7d_weekend_days: int
    next_7d_event_days: int
    next_7d_snap_days: int
    category_weekend_lift_pct: float
    category_holiday_lift_pct: float
    category_snap_lift_pct: float
    category_latest_mom_pct: float
    replenishment_formula_breakdown: str


class SQLQueryRequest(BaseModel):
    """Dynamic read-only SQL query payload guarded by sqlglot AST firewall."""

    sql: str = Field(
        description=(
            "Single DuckDB SELECT or WITH...SELECT query targeting only "
            "ads_*, dws_*, or ops_* tables."
        )
    )
    max_rows: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Requested max rows (server clamps to SQL_MAX_ROWS=500).",
    )


class SQLQueryResponse(BaseModel):
    """Result set returned by the read-only SQL gateway."""

    columns: list[str]
    row_count: int
    rows: list[dict[str, Any]]
    referenced_tables: list[str]
    executed_sql: str
