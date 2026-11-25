# LakePulse-AI 工程开发规范（AGENTS.md）

本文件定义了 LakePulse-AI 项目在数据建模、特征工程、模型推理与 API 安全网关中的强制性工程约束，所有新增代码、SQL 模型与脚本均须遵守。

---

## 1. 运行环境与 Python 代码规范

1. **Python 与 Pandas 版本**：统一使用 **Python 3.11** 与 **`pandas >= 2.1.0, < 3.0.0`**（严禁在 Python 3.13+ 或 Pandas 3.x 环境中运行 PySpark 3.5.x）。
2. **类型注解与代码风格**：
   - 所有公共函数、类方法及配置模型必须具备完整类型注解（Type Hints）与简明 docstring。
   - 代码须通过 `ruff check src tests scripts` 静态检查，零告警。
3. **跨平台与 JDK 17 自动隔离绑定**：
   - PySpark 3.5 内置 Arrow 12（`@pandas_udf` 与 Arrow `toPandas()`）需运行于 **OpenJDK 17**（在 `lakepulse` Conda 环境中隔离安装，且保持 `pandas < 3.0`）。
   - 获取 Spark 会话必须统一调用 `lakepulse.spark.session.build_spark_session`，禁止在业务模块中零散 `SparkSession.builder` 绕过 Conda JDK 17 `JAVA_HOME` 自动接管与 JVM `--add-opens` / `-Djava.security.manager=allow` 注入。
   - 读写 Parquet 必须统一调用 `lakepulse.spark.io.write_parquet` 与 `read_parquet`，确保本地 Windows（无 `winutils.exe`）与集群 HDFS 模式均可无感知切换。

---

## 2. 数仓分层与命名规范

表与路径统一按以下五类前缀分层存放，严禁跨层混乱命名：

| 层级前缀 | 职责定位 | 典型表名 |
| :--- | :--- | :--- |
| `ods_` | 原始贴源层（超宽表逆透视展开、原始口径对齐，不做复杂业务聚合） | `ods_sales_long`, `ods_calendar`, `ods_sell_prices` |
| `dwd_` | 明细清洗层（关联日历/售价、空值与负值治理、统一主键粒度 `(store_id, item_id, date)`） | `dwd_daily_item_sales` |
| `dws_` | 汇总与特征宽表层（ABC 分层画像、多尺度滑窗统计特征） | `dws_item_abc_profile`, `dws_item_rolling_features` |
| `ads_` | 应用决策层（面向采购补货、库存监控与 Agent 查询的最终结果表） | `ads_replenishment_plan` |
| `ops_` | 运维监控与审计层（记录每次 ETL/建模/推理任务的跑批状态与数据健康度） | `ops_pipeline_audit` |

---

## 3. 机器学习防数据穿越铁律（No Data Leakage）

1. **严格时间边界**：对于预测基准日 $T$（预测区间为 $[T+1, T+7]$），历史统计特征（如 7 天均销、28 天均销、波动标准差、历史均价）**只能使用 $\le T$ 的历史观测值**（在 Spark Window 中对应 `ROWS BETWEEN 27 PRECEDING AND CURRENT ROW` 或严格滞后列）。
2. **未来已知排期白名单**：仅允许使用业务上在 $T$ 日前已知的未来排期字段（如未来 7 天的法定节假日天数、周末天数、已知促销/消费券日历标记及预设零售价），**严禁将未来实际销量或未来全局均值泄露进训练/推理特征**。
3. **时间序列切分**：训练、验证与测试集必须严格按日期先后顺序切分（Time-based Split），禁止随机打乱（Random Split）造成未来信息穿越。

---

## 4. FastAPI 数据服务安全红线

1. **SQL 只读防火墙（基于 `sqlglot` AST）**：
   - 所有对外暴露的动态 SQL 查询入口必须先经 `sqlglot` 解析语法树（AST）。
   - 仅允许单条 `SELECT`（含 `WITH ... SELECT` CTE）语句执行；任何包含 `INSERT`、`UPDATE`、`DELETE`、`DROP`、`ALTER`、`CREATE`、`ATTACH`、`COPY`、`PRAGMA` 或多语句分号注入的请求必须直接拒绝（HTTP 400 / 403）。
2. **表白名单与结果集限制**：
   - 只允许查询 `ads_*`、`dws_*`、`ops_*` 白名单视图/表，禁止直接暴露未脱敏或原始底表。
   - 强制注入或校验结果集上限（默认 `SQL_MAX_ROWS=500`）与查询超时保护，防止大表全扫描拖垮查询服务。
