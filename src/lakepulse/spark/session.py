"""SparkSession builder with Conda JDK 17 auto-binding and dual-mode support."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession

from lakepulse.config import AppSettings, get_settings

SPARK_JVM_OPTIONS: tuple[str, ...] = (
    "-Djava.security.manager=allow",
    "-Dio.netty.tryReflectionSetAccessible=true",
    "--add-opens=java.base/java.lang=ALL-UNNAMED",
    "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
    "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
    "--add-opens=java.base/java.io=ALL-UNNAMED",
    "--add-opens=java.base/java.net=ALL-UNNAMED",
    "--add-opens=java.base/java.nio=ALL-UNNAMED",
    "--add-opens=java.base/java.util=ALL-UNNAMED",
    "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
    "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
    "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
    "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
    "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
)


def _configure_conda_java_home() -> Path | None:
    """Bind JAVA_HOME and PATH to the active Python env's JDK 17 if installed."""
    prefix = Path(sys.prefix)
    candidates = (
        prefix / "Library" / "lib" / "jvm",
        prefix / "Library",
        prefix / "lib" / "jvm",
    )
    java_exe = "java.exe" if os.name == "nt" else "java"

    for candidate in candidates:
        bin_dir = candidate / "bin"
        if (bin_dir / java_exe).is_file():
            os.environ["JAVA_HOME"] = str(candidate)
            current_path = os.environ.get("PATH", "")
            path_entries = [p for p in current_path.split(os.pathsep) if p]
            bin_str = str(bin_dir)
            if not path_entries or Path(path_entries[0]) != bin_dir:
                filtered = [p for p in path_entries if Path(p) != bin_dir]
                os.environ["PATH"] = os.pathsep.join([bin_str, *filtered])
            return candidate
    return None


def _merge_java_options(existing: str, required: tuple[str, ...]) -> str:
    """Append required JVM flags if not already present."""
    tokens = existing.split() if existing else []
    token_set = set(tokens)
    for flag in required:
        if flag not in token_set:
            tokens.append(flag)
            token_set.add(flag)
    return " ".join(tokens)


def configure_jvm_env() -> str:
    """Configure JDK, Python executables, and JVM options before Py4J starts."""
    _configure_conda_java_home()
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    merged_opts = _merge_java_options(
        os.environ.get("JDK_JAVA_OPTIONS", ""),
        SPARK_JVM_OPTIONS,
    )
    os.environ["JDK_JAVA_OPTIONS"] = merged_opts
    return merged_opts


def build_spark_session(
    app_name: str = "LakePulse-AI",
    settings: AppSettings | None = None,
) -> SparkSession:
    """Build or reuse a configured SparkSession for local or cluster mode."""
    jvm_opts = configure_jvm_env()
    cfg = settings or get_settings()
    spark_cfg = cfg.yaml_config.get("spark", {})

    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.driver.extraJavaOptions", jvm_opts)
        .config("spark.executor.extraJavaOptions", jvm_opts)
        .config(
            "spark.sql.execution.arrow.pyspark.enabled",
            str(spark_cfg.get("arrow_enabled", True)).lower(),
        )
    )

    if cfg.mode == "cluster":
        builder = (
            builder.master(cfg.spark_master_url)
            .config("spark.hadoop.fs.defaultFS", cfg.hdfs_namenode_uri)
            .config("spark.sql.catalogImplementation", "hive")
            .config("spark.hadoop.hive.metastore.uris", cfg.hive_metastore_uri)
            .enableHiveSupport()
        )
    else:
        builder = (
            builder.master(str(spark_cfg.get("local_master", "local[2]")))
            .config("spark.driver.memory", str(spark_cfg.get("driver_memory", "2g")))
            .config("spark.sql.shuffle.partitions", str(spark_cfg.get("shuffle_partitions", 4)))
            .config("spark.ui.enabled", str(spark_cfg.get("ui_enabled", False)).lower())
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
        )

    return builder.getOrCreate()
