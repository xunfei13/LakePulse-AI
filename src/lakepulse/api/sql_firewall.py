"""AST-based read-only SQL firewall powered by sqlglot."""

from __future__ import annotations

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

ALLOWED_TABLE_PREFIXES: tuple[str, ...] = ("ads_", "dws_", "ops_")

ALLOWED_ROOT_TYPES: frozenset[str] = frozenset({"Select", "Union"})

FORBIDDEN_AST_NODE_NAMES: frozenset[str] = frozenset(
    {
        "Insert",
        "Update",
        "Delete",
        "Drop",
        "Alter",
        "Create",
        "Copy",
        "Command",
        "Pragma",
        "Set",
        "Use",
        "Transaction",
        "Commit",
        "Rollback",
        "Attach",
        "Detach",
        "Install",
        "Load",
        "LoadData",
        "TruncateTable",
        "Merge",
        "Grant",
        "Revoke",
        "Kill",
    }
)

FORBIDDEN_TABLE_FUNCTIONS: frozenset[str] = frozenset(
    {
        "read_parquet",
        "parquet_scan",
        "read_csv",
        "read_csv_auto",
        "csv_scan",
        "read_json",
        "read_json_auto",
        "read_ndjson",
        "read_ndjson_auto",
        "read_text",
        "read_blob",
        "glob",
        "sqlite_scan",
        "postgres_scan",
        "mysql_scan",
        "iceberg_scan",
        "delta_scan",
        "http_get",
        "query",
        "query_table",
    }
)


class SQLSyntaxValidationError(ValueError):
    """Raised when incoming SQL is empty or cannot be parsed."""


class SQLSecurityViolationError(PermissionError):
    """Raised when SQL violates read-only, single-statement, or table whitelist rules."""


def _extract_literal_limit(stmt: exp.Expression) -> int | None:
    """Return integer LIMIT if present on outer query, else None."""
    limit_expr = stmt.args.get("limit")
    if limit_expr is None:
        return None
    inner = getattr(limit_expr, "expression", None)
    if isinstance(inner, exp.Literal) and inner.is_int:
        return int(inner.this)
    return None


def validate_and_rewrite_sql(sql: str, max_rows: int = 500) -> tuple[str, list[str]]:
    """Validate SQL AST for read-only safety and enforce a maximum row LIMIT."""
    if not sql or not sql.strip():
        raise SQLSyntaxValidationError("SQL query must not be empty.")
    if max_rows <= 0:
        raise ValueError("max_rows must be a positive integer.")

    try:
        statements = [
            s for s in sqlglot.parse(sql, read="duckdb") if s is not None
        ]
    except ParseError as exc:
        raise SQLSyntaxValidationError(f"Invalid DuckDB SQL syntax: {exc}") from exc

    if len(statements) == 0:
        raise SQLSyntaxValidationError("SQL query must contain one SELECT statement.")
    if len(statements) > 1:
        raise SQLSecurityViolationError(
            "Multiple SQL statements are forbidden; only a single SELECT query is allowed."
        )

    stmt = statements[0]
    root_name = type(stmt).__name__
    if root_name not in ALLOWED_ROOT_TYPES:
        raise SQLSecurityViolationError(
            f"Statement type '{root_name}' is forbidden; only SELECT queries are allowed."
        )

    for node in stmt.walk():
        node_name = type(node).__name__
        if node_name in FORBIDDEN_AST_NODE_NAMES:
            raise SQLSecurityViolationError(
                f"Forbidden SQL operation '{node_name}' detected in query AST."
            )
        if isinstance(node, exp.Func):
            fn_name = (node.sql_name() or getattr(node, "name", "") or "").strip().lower()
            if fn_name in FORBIDDEN_TABLE_FUNCTIONS or fn_name.startswith(
                ("duckdb_", "pragma_")
            ):
                raise SQLSecurityViolationError(
                    f"Function '{fn_name}' is blocked by SQL read-only firewall."
                )

    cte_names = {
        cte.alias_or_name.strip().lower()
        for cte in stmt.find_all(exp.CTE)
        if cte.alias_or_name
    }

    referenced_tables: set[str] = set()
    for tbl in stmt.find_all(exp.Table):
        catalog_name = (tbl.catalog or "").strip().lower()
        db_name = (tbl.db or "").strip().lower()
        table_name = (tbl.name or "").strip().lower()

        if catalog_name or (db_name and db_name != "main"):
            schema_label = catalog_name or db_name
            raise SQLSecurityViolationError(
                f"Access to external or system schema '{schema_label}' is forbidden."
            )
        if not table_name:
            raise SQLSecurityViolationError(
                "Anonymous or table-valued function sources are forbidden."
            )

        if table_name not in cte_names:
            if not table_name.startswith(ALLOWED_TABLE_PREFIXES):
                raise SQLSecurityViolationError(
                    f"Table '{table_name}' is not in allowed prefixes {ALLOWED_TABLE_PREFIXES}."
                )
            referenced_tables.add(table_name)

    outer_limit = _extract_literal_limit(stmt)
    effective_limit = (
        min(max(1, outer_limit), max_rows) if outer_limit is not None else max_rows
    )

    clean_sql = stmt.sql(dialect="duckdb").rstrip().rstrip(";")
    rewritten_sql = f"SELECT * FROM ({clean_sql}) AS _lakepulse_q LIMIT {effective_limit}"
    return rewritten_sql, sorted(referenced_tables)
