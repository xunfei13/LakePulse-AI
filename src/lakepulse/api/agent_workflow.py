"""OpenAPI schema exporter and offline 4-step replenishment Agent closed-loop runner."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from lakepulse.api.app import create_app
from lakepulse.config import AppSettings, get_settings


def export_dify_openapi_schema(
    output_path: Path,
    server_url: str = "http://host.docker.internal:8000",
    settings: AppSettings | None = None,
) -> Path:
    """Export FastAPI OpenAPI JSON configured with server URL for Dify custom tools."""
    app = create_app(settings)
    schema = app.openapi()
    schema["servers"] = [
        {"url": server_url, "description": "LakePulse-AI FastAPI Data Gateway"}
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def _format_workorder_markdown(
    *,
    store_id: str,
    abc_class: str,
    max_budget: float,
    ops_health: dict[str, Any],
    alerts: dict[str, Any],
    attributions: list[dict[str, Any]],
    category_summary: list[dict[str, Any]],
) -> str:
    """Render an executive replenishment approval workorder in Markdown."""
    deployed_metric = next(
        (m for m in ops_health.get("test_model_metrics", []) if m.get("deployed_for_inference")),
        None,
    )
    if deployed_metric:
        model_line = (
            f"{deployed_metric['model_name']} | "
            f"TEST WAPE={deployed_metric['wape_pct']:.2f}%, "
            f"R2={deployed_metric['r2_score']:.3f}"
        )
    else:
        model_line = "N/A"

    lines = [
        f"# LakePulse-AI 智能补货审批工单（门店：{store_id} / 品级：{abc_class} 类）",
        "",
        "## 1. 上游湖仓与模型健康巡检",
        f"- **跑批整体状态**：`{ops_health['overall_status']}`"
        f"（FAILED 表：{ops_health['failed_tables'] or '无'}；"
        f"WARN 表：{ops_health['warn_tables'] or '无'}）",
        f"- **线上推理模型**：{model_line}",
        "",
        "## 2. 预算约束内高优补货清单",
        f"- **筛选条件**：门店 `{store_id}`，ABC 等级 `{abc_class}`，预算上限 `${max_budget:,.2f}`",
        f"- **匹配高危总数**：{alerts['total_matched']} 款 | "
        f"**预算内选中**：{alerts['returned_count']} 款 | "
        f"**建议采购总量**：{alerts['total_recommended_order_qty']} 件 | "
        f"**预估采购总额**：`${alerts['total_estimated_order_amount']:,.2f}`",
        "",
        (
            "| 商品 ID | 品类 | 未来7天预测 | 现存可用 | 安全缓冲 | "
            "建议补货 | 覆盖天数 | 风险 | 预估金额($) |"
        ),
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for item in alerts.get("items", []):
        lines.append(
            f"| {item['item_id']} | {item['cat_id']} | "
            f"{item['pred_next_7d_sales']:.1f} | {item['current_available_stock']} | "
            f"{item['safety_stock_qty']:.1f} | **{item['recommended_order_qty']}** | "
            f"{item['coverage_days']:.1f}d | {item['stockout_risk_level']} | "
            f"{item['estimated_order_amount']:.2f} |"
        )

    lines.extend(["", "## 3. 重点单品需求涨跌归因拆解"])
    for attr in attributions:
        lines.append(
            f"- **{attr['item_id']}**（标签：`{attr['key_driver_tags']}`）："
            f"7天均销 `{attr['sales_ma_7']:.2f}` vs 28天均销 `{attr['sales_ma_28']:.2f}`"
            f"（趋势比 `{attr['sales_trend_ratio']:.2f}`），"
            f"当前折扣率 `{attr['price_discount_rate'] * 100:.1f}%`，"
            f"未来7天含 `{attr['next_7d_event_days']}` 天节假日 / "
            f"`{attr['next_7d_snap_days']}` 天 SNAP 窗口；"
            f"口径计算：`{attr['replenishment_formula_breakdown']}`"
        )

    lines.extend(
        [
            "",
            "## 4. 门店全品类风险分布核验（经 `sqlglot` 只读防火墙 SQL 查询）",
            "| 品类 | 风险等级 | SKU 数 | 建议补货件数 | 预估采购额($) |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
    )
    for row in category_summary:
        lines.append(
            f"| {row['cat_id']} | {row['stockout_risk_level']} | "
            f"{row['sku_cnt']} | {row['total_order_qty']} | {row['total_est_cost']:.2f} |"
        )

    return "\n".join(lines)


def run_replenishment_agent_closed_loop(
    settings: AppSettings | None = None,
    *,
    store_id: str = "CA_1",
    abc_class: str = "A",
    max_budget: float = 1500.0,
    top_k_explain: int = 3,
) -> dict[str, Any]:
    """Execute the 4-step Agent replenishment workflow end-to-end against FastAPI gateway."""
    if not re.fullmatch(r"[A-Za-z0-9_]+", store_id):
        raise ValueError(f"Invalid store_id format: {store_id}")

    cfg = settings or get_settings()
    app = create_app(cfg)

    with TestClient(app) as client:
        # Step 1: Inspect pipeline & model health
        health_resp = client.get("/api/v1/ops/health-summary")
        if health_resp.status_code != 200:
            raise RuntimeError(f"Health summary failed: {health_resp.text}")
        ops_health = health_resp.json()
        if ops_health["overall_status"] == "FAILED":
            raise RuntimeError(
                f"Upstream pipeline health check failed: {ops_health['failed_tables']}"
            )

        # Step 2: Query budget-constrained high-risk replenishment alerts
        alerts_resp = client.get(
            "/api/v1/replenishment/alerts",
            params={
                "store_id": store_id,
                "abc_class": abc_class,
                "risk_level": "HIGH",
                "only_need_order": True,
                "max_budget": max_budget,
                "limit": 20,
            },
        )
        if alerts_resp.status_code != 200:
            raise RuntimeError(f"Alerts query failed: {alerts_resp.text}")
        alerts_data = alerts_resp.json()

        if alerts_data["returned_count"] == 0:
            alerts_resp = client.get(
                "/api/v1/replenishment/alerts",
                params={
                    "store_id": store_id,
                    "abc_class": abc_class,
                    "only_need_order": True,
                    "max_budget": max_budget,
                    "limit": 20,
                },
            )
            if alerts_resp.status_code != 200:
                raise RuntimeError(f"Fallback alerts query failed: {alerts_resp.text}")
            alerts_data = alerts_resp.json()

        # Step 3: Explain top-K urgent SKUs and cross-verify via read-only SQL firewall
        attributions: list[dict[str, Any]] = []
        for item in alerts_data["items"][:top_k_explain]:
            attr_resp = client.get(
                f"/api/v1/replenishment/sku-attribution/{item['store_id']}/{item['item_id']}"
            )
            if attr_resp.status_code == 200:
                attributions.append(attr_resp.json())

        sql_resp = client.post(
            "/api/v1/sql/query",
            json={
                "sql": (
                    "SELECT cat_id, stockout_risk_level, COUNT(*) AS sku_cnt, "
                    "SUM(recommended_order_qty) AS total_order_qty, "
                    "ROUND(SUM(estimated_order_amount), 2) AS total_est_cost "
                    "FROM ads_replenishment_plan "
                    f"WHERE store_id = '{store_id}' "
                    "GROUP BY cat_id, stockout_risk_level "
                    "ORDER BY cat_id, stockout_risk_level"
                ),
                "max_rows": 100,
            },
        )
        if sql_resp.status_code != 200:
            raise RuntimeError(f"SQL verification query failed: {sql_resp.text}")
        sql_data = sql_resp.json()

    # Step 4: Render actionable procurement workorder
    workorder_md = _format_workorder_markdown(
        store_id=store_id,
        abc_class=abc_class,
        max_budget=max_budget,
        ops_health=ops_health,
        alerts=alerts_data,
        attributions=attributions,
        category_summary=sql_data["rows"],
    )

    return {
        "store_id": store_id,
        "abc_class": abc_class,
        "max_budget": max_budget,
        "ops_health": ops_health,
        "alerts": alerts_data,
        "attributions": attributions,
        "category_risk_summary": sql_data["rows"],
        "workorder_markdown": workorder_md,
    }
