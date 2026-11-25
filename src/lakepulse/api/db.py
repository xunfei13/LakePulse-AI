"""Read-only DuckDB query executor with external access lock and timeout interruption."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import duckdb

from lakepulse.api.sql_firewall import ALLOWED_TABLE_PREFIXES
from lakepulse.config import AppSettings


class SQLQueryTimeoutError(TimeoutError):
    """Raised when a DuckDB query exceeds the configured timeout threshold."""


def _to_json_value(value: Any) -> Any:
    """Convert DuckDB native Python values into JSON-serializable primitives."""
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def execute_readonly_query(
    settings: AppSettings,
    sql: str,
    params: list[Any] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Execute a SQL query on a read-only DuckDB connection with external access disabled."""
    if not settings.duckdb_path.is_file():
        raise FileNotFoundError(
            f"DuckDB lakehouse file not found: {settings.duckdb_path}. Run Stage 1-3 first."
        )

    holder: dict[str, duckdb.DuckDBPyConnection] = {}

    def _run() -> tuple[list[str], list[dict[str, Any]]]:
        con = duckdb.connect(str(settings.duckdb_path), read_only=True)
        holder["con"] = con
        try:
            con.execute("SET enable_external_access = false")
            cursor = con.execute(sql, params or [])
            columns = [desc[0] for desc in (cursor.description or [])]
            raw_rows = cursor.fetchall()
            rows = [
                {col: _to_json_value(val) for col, val in zip(columns, row, strict=True)}
                for row in raw_rows
            ]
            return columns, rows
        finally:
            con.close()

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_run)
    try:
        return future.result(timeout=float(settings.sql_timeout_sec))
    except FuturesTimeoutError as exc:
        con = holder.get("con")
        if con is not None:
            try:
                con.interrupt()
            except Exception:
                pass
        raise SQLQueryTimeoutError(
            f"Query exceeded timeout of {settings.sql_timeout_sec}s and was interrupted."
        ) from exc
    finally:
        executor.shutdown(wait=False)


def list_whitelisted_tables(settings: AppSettings) -> list[str]:
    """List existing whitelisted tables in DuckDB."""
    if not settings.duckdb_path.is_file():
        return []
    _, rows = execute_readonly_query(settings, "SHOW TABLES")
    tables = [
        str(r["name"])
        for r in rows
        if str(r.get("name", "")).lower().startswith(ALLOWED_TABLE_PREFIXES)
    ]
    return sorted(tables)
