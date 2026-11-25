"""CLI demo exporting Dify OpenAPI tools and running the 4-step Agent replenishment loop."""

from __future__ import annotations

import argparse

from lakepulse.api.agent_workflow import (
    export_dify_openapi_schema,
    run_replenishment_agent_closed_loop,
)
from lakepulse.config import get_settings


def main() -> int:
    """Export OpenAPI JSON for Dify and generate an end-to-end replenishment workorder."""
    parser = argparse.ArgumentParser(
        description="Run LakePulse-AI Stage 4 Dify OpenAPI export and Agent closed loop."
    )
    parser.add_argument("--store-id", default="CA_1", help="Target store ID (e.g. CA_1).")
    parser.add_argument("--abc-class", default="A", help="ABC value class (A, B, or C).")
    parser.add_argument(
        "--max-budget",
        type=float,
        default=1500.0,
        help="Maximum procurement budget cap ($).",
    )
    parser.add_argument(
        "--server-url",
        default="http://host.docker.internal:8000",
        help="Gateway base URL written into docs/dify/openapi_tools.json.",
    )
    args = parser.parse_args()

    cfg = get_settings()
    openapi_path = cfg.project_root / "docs" / "dify" / "openapi_tools.json"
    export_dify_openapi_schema(openapi_path, server_url=args.server_url, settings=cfg)
    print(f"[Stage 4] Exported Dify OpenAPI tool schema -> {openapi_path}")

    result = run_replenishment_agent_closed_loop(
        cfg,
        store_id=args.store_id,
        abc_class=args.abc_class,
        max_budget=args.max_budget,
    )

    report_path = cfg.project_root / "artifacts" / "reports" / "latest_replenishment_workorder.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(result["workorder_markdown"], encoding="utf-8")
    print(f"[Stage 4] Saved Agent replenishment workorder -> {report_path}\n")
    print(result["workorder_markdown"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
