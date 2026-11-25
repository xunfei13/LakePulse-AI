"""Pipeline execution audit tracking for ops_pipeline_audit."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

from lakepulse.config import AppSettings, get_settings
from lakepulse.spark.io import write_parquet
from lakepulse.spark.session import build_spark_session

VALID_AUDIT_STATUSES = frozenset({"SUCCESS", "WARN", "FAILED"})

AUDIT_SPARK_SCHEMA = StructType(
    [
        StructField("run_id", StringType(), nullable=False),
        StructField("pipeline_stage", StringType(), nullable=False),
        StructField("table_name", StringType(), nullable=False),
        StructField("started_at", StringType(), nullable=False),
        StructField("finished_at", StringType(), nullable=False),
        StructField("duration_sec", DoubleType(), nullable=False),
        StructField("input_rows", LongType(), nullable=False),
        StructField("output_rows", LongType(), nullable=False),
        StructField("null_rate", DoubleType(), nullable=False),
        StructField("status", StringType(), nullable=False),
        StructField("message", StringType(), nullable=False),
    ]
)

AUDIT_ARROW_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("pipeline_stage", pa.string()),
        ("table_name", pa.string()),
        ("started_at", pa.string()),
        ("finished_at", pa.string()),
        ("duration_sec", pa.float64()),
        ("input_rows", pa.int64()),
        ("output_rows", pa.int64()),
        ("null_rate", pa.float64()),
        ("status", pa.string()),
        ("message", pa.string()),
    ]
)


@dataclass(frozen=True)
class AuditRecord:
    """Single table or stage execution record persisted to ops_pipeline_audit."""

    run_id: str
    pipeline_stage: str
    table_name: str
    started_at: str
    finished_at: str
    duration_sec: float
    input_rows: int
    output_rows: int
    null_rate: float
    status: str
    message: str

    def __post_init__(self) -> None:
        if self.status not in VALID_AUDIT_STATUSES:
            raise ValueError(
                f"Invalid audit status '{self.status}'. Expected one of {VALID_AUDIT_STATUSES}."
            )


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def generate_run_id(prefix: str = "run") -> str:
    """Generate a unique sortable batch run identifier."""
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}_{uuid.uuid4().hex[:8]}"


def append_audit_records_spark(
    spark: SparkSession,
    settings: AppSettings,
    records: list[AuditRecord],
) -> None:
    """Append audit records to ops_pipeline_audit Parquet store via Spark."""
    if not records:
        return
    rows = [asdict(r) for r in records]
    df = spark.createDataFrame(rows, schema=AUDIT_SPARK_SCHEMA)
    audit_uri = settings.resolve_table_uri("ops", "ops_pipeline_audit")
    write_parquet(df, audit_uri, mode="append")


def append_audit_records_local(
    settings: AppSettings,
    records: list[AuditRecord],
) -> None:
    """Append audit records without starting Spark in local mode (falls back in cluster)."""
    if not records:
        return
    if settings.mode == "cluster":
        spark = build_spark_session("LakePulse-Audit-Logger", settings)
        append_audit_records_spark(spark, settings, records)
        return

    audit_dir = Path(settings.resolve_table_uri("ops", "ops_pipeline_audit"))
    audit_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([asdict(r) for r in records], schema=AUDIT_ARROW_SCHEMA)
    part_path = audit_dir / f"part-{uuid.uuid4().hex}.parquet"
    pq.write_table(table, part_path)


def sync_audit_table_to_duckdb(settings: AppSettings | None = None) -> int:
    """Materialize ops_pipeline_audit Parquet logs into DuckDB and return row count."""
    cfg = settings or get_settings()
    audit_dir = cfg.lakehouse_dir / "ops" / "ops_pipeline_audit"
    if not audit_dir.exists() or not any(audit_dir.rglob("*.parquet")):
        return 0

    cfg.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    glob_posix = f"{audit_dir.resolve().as_posix()}/**/*.parquet"
    con = duckdb.connect(str(cfg.duckdb_path))
    try:
        con.execute(
            f"""
            CREATE OR REPLACE TABLE ops_pipeline_audit AS
            SELECT * FROM read_parquet('{glob_posix}')
            """
        )
        row = con.execute("SELECT COUNT(*) FROM ops_pipeline_audit").fetchone()
        return int(row[0]) if row is not None else 0
    finally:
        con.close()
