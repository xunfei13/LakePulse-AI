"""Dual-mode configuration management and lakehouse URI resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AppSettings(BaseSettings):
    """Runtime settings combining environment variables, .env, and settings.yaml."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    mode: Literal["local", "cluster"] = Field(default="local", validation_alias="LAKEPULSE_MODE")
    project_root: Path = Field(
        default=DEFAULT_PROJECT_ROOT, validation_alias="LAKEPULSE_PROJECT_ROOT"
    )
    data_dir_raw: Path = Field(default=Path("data"), validation_alias="LAKEPULSE_DATA_DIR")
    duckdb_path_raw: Path = Field(
        default=Path("data/lakehouse/lakepulse.duckdb"),
        validation_alias="LAKEPULSE_DUCKDB_PATH",
    )
    models_dir_raw: Path = Field(
        default=Path("artifacts/models"),
        validation_alias="LAKEPULSE_MODELS_DIR",
    )

    spark_master_url: str = Field(
        default="spark://localhost:7077", validation_alias="SPARK_MASTER_URL"
    )
    hdfs_namenode_uri: str = Field(
        default="hdfs://localhost:9000", validation_alias="HDFS_NAMENODE_URI"
    )
    hive_metastore_uri: str = Field(
        default="thrift://localhost:9083", validation_alias="HIVE_METASTORE_URI"
    )

    api_host: str = Field(default="127.0.0.1", validation_alias="API_HOST")
    api_port: int = Field(default=8000, validation_alias="API_PORT")
    sql_max_rows: int = Field(default=500, validation_alias="SQL_MAX_ROWS")
    sql_timeout_sec: int = Field(default=10, validation_alias="SQL_TIMEOUT_SEC")

    yaml_config: dict[str, Any] = Field(default_factory=dict)

    def _resolve_path(self, path: Path) -> Path:
        """Resolve relative paths against project_root."""
        return path if path.is_absolute() else (self.project_root / path).resolve()

    @property
    def data_dir(self) -> Path:
        """Absolute root directory for raw and processed data."""
        return self._resolve_path(self.data_dir_raw)

    @property
    def lakehouse_dir(self) -> Path:
        """Absolute directory for local lakehouse Parquet layers."""
        return self.data_dir / "lakehouse"

    @property
    def duckdb_path(self) -> Path:
        """Absolute path to the DuckDB database file."""
        return self._resolve_path(self.duckdb_path_raw)

    @property
    def models_dir(self) -> Path:
        """Absolute directory for serialized ML/DL model artifacts."""
        return self._resolve_path(self.models_dir_raw)

    def resolve_table_uri(self, layer: str, table_name: str) -> str:
        """Resolve storage URI for a lakehouse table under local or cluster mode."""
        clean_layer = layer.strip().lower()
        clean_table = table_name.strip()
        if not clean_layer or not clean_table:
            raise ValueError("Both layer and table_name must be non-empty strings.")

        if self.mode == "cluster":
            base_uri = self.hdfs_namenode_uri.rstrip("/")
            return f"{base_uri}/lakepulse/{clean_layer}/{clean_table}"

        return str((self.lakehouse_dir / clean_layer / clean_table).resolve())


def load_yaml_settings(config_path: Path) -> dict[str, Any]:
    """Load structured YAML configuration if present."""
    if not config_path.is_file():
        return {}
    with config_path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    return loaded if isinstance(loaded, dict) else {}


def get_settings(mode: Literal["local", "cluster"] | None = None) -> AppSettings:
    """Create an AppSettings instance with configs/settings.yaml merged in."""
    settings = AppSettings()
    if mode is not None:
        settings.mode = mode
    yaml_path = settings.project_root / "configs" / "settings.yaml"
    settings.yaml_config = load_yaml_settings(yaml_path)
    return settings
