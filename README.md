# LakePulse-AI：基于湖仓一体的零售销量预测与智能补货决策平台

> 以 **Walmart M5** 真实千万级零售交易、售价与日历数据为底座，构建 **“分布式数据清洗与监控 → 商业分析与特征工程 → TensorFlow 深度学习批量预测 → SQL 只读安全网关与 AI Agent 补货闭环”** 的端到端湖仓一体智能决策系统。

---

## 一、项目核心目的

1. **业务价值**：打破连锁零售仓储“凭经验拍脑袋备货，畅销品断货丢单、滞销品积压占款”的困局，打通从多源原始流水治理、ABC 商品价值分层、未来 7 天需求预测到分仓建议补货工单生成的决策闭环。
2. **工程定位**：以单一真实主线完整串联大数据工程（`Hadoop / Hive Metastore / PySpark / Parquet / dbt`）、机器学习与深度学习实战（`LightGBM Baseline` + `TensorFlow Wide & Deep` + `@pandas_udf` 分布式推理）以及数据服务安全治理（`DuckDB + FastAPI + sqlglot + Dify Agent`），并支持 **Local 轻量模式** 与 **16GB 内存 Docker 伪分布式集群模式** 一键切换。

---

## 二、解决的四大核心问题

| 核心痛点 | 工程与业务解法 | 对应技术组件 |
| :--- | :--- | :--- |
| **1. 多源畸形数据清洗与日常跑批监控** | 针对 M5“1900+ 天横排列名”超宽表与周度价格粒度不一致问题，使用 PySpark 逆透视展开并关联对齐；依托 `dbt-core` 落地分层建模与质量门禁，内建 `ops_pipeline_audit` 跑批审计表监控耗时、行数波动与空值率。 | `HDFS` / `PySpark` / `Parquet` / `dbt-core` |
| **2. 主次不分与“预测销量 ≠ 建议补货”** | 基于二八法则划分 `ABC` 商品等级，量化周末/节假日同环比及价格折扣拉动；结合差异化安全缓冲天数（A 类 3 天、B 类 2 天、C 类 1 天）与当前可用库存核算安全补货量。 | `Spark SQL` / `DuckDB` / 零售库存模型 |
| **3. 多因素需求拟合与海量商品推理瓶颈** | 用 Spark Window 构造无数据穿越的二维特征宽表，对比统计基准与 `LightGBM` 后，训练 **TensorFlow Keras `Wide & Deep`** 回归模型预测未来 7 天销量，并通过 PySpark `@pandas_udf` 在 Executor 分区级加载权重完成批量推理。 | `Spark Window` / `LightGBM` / `TensorFlow 2.x` / `@pandas_udf` |
| **4. 业务查数门槛高与大模型直连安全隐患** | DuckDB 原地挂载 Parquet 提供毫秒级 OLAP 查询；FastAPI 内置 `sqlglot` AST 只读防火墙拦截写操作与慢查询并暴露 OpenAPI 工具集，联动 Dify Agent 与指标 RAG 知识库输出可解释的分仓采购建议。 | `DuckDB` / `FastAPI` / `sqlglot` / `Dify` |

---

## 三、端到端系统架构

```mermaid
flowchart LR
    RAW["Walmart M5 原始数据\n(sales / calendar / sell_prices)"] --> ETL["PySpark 分布式 ETL\n超宽表逆透视 + 清洗对齐"]
    ETL --> LAKE["Lakehouse Parquet 分区存储\n(Local FS / HDFS + Hive Metastore)"]
    LAKE --> DBT["dbt-core 数仓建模与监控\n(ODS -> DWD -> DWS -> OPS 审计)"]
    DBT --> FEAT["PySpark Window 特征工程\nABC分层 + 滑窗均值 + 折扣率"]
    FEAT --> TRAIN["模型训练与对比\nNaive / LightGBM / TF Wide & Deep"]
    TRAIN --> INFER["PySpark @pandas_udf 分布式推理\n+ 安全缓冲补货公式"]
    INFER --> ADS["ADS 补货决策宽表\n(ads_replenishment_plan)"]
    ADS --> DUCK["DuckDB 毫秒级 OLAP 挂载"]
    DUCK --> GW["FastAPI 数据安全网关\n(sqlglot AST 只读防火墙 + OpenAPI)"]
    GW --> AGENT["Dify 智能补货 Agent\n(+ 指标口径 RAG 知识库)"]
```

---

## 四、双运行模式说明

项目通过环境变量 `LAKEPULSE_MODE` 与 `configs/settings.yaml` 无缝支持双模式运行：

1. **Local 轻量开发模式（默认，`LAKEPULSE_MODE=local`）**
   - 计算与存储：`PySpark local[2]` + 本地目录 `data/lakehouse/` + `DuckDB` + `dbt-duckdb`；
   - 默认按 `configs/settings.yaml` 聚焦抽样门店（如 `CA_1`、`CA_2`），可在 16GB 内存笔记本上秒级调试与端到端跑通。
2. **Docker 伪分布式集群模式（`LAKEPULSE_MODE=cluster`）**
   - 容器编排：[docker/docker-compose.yml](docker/docker-compose.yml) 拉起 `HDFS NameNode`、`HDFS DataNode`、`Hive Metastore`、`Spark Master` 与 `Spark Worker`（总内存上限约 6.5 GB，适配 16GB 宿主机）；
   - 存储路径自动切换为 `hdfs://localhost:9000/lakepulse/<layer>/<table>`，无需修改业务代码。

---

## 五、目录结构

```text
LakePulse-AI/
├── AGENTS.md                       # 工程开发规范与安全红线
├── Makefile                        # Linux/macOS/WSL 统一任务入口
├── make.ps1                        # Windows PowerShell 统一任务入口
├── pyproject.toml                  # Python 3.11 依赖与打包配置
├── .env.example                    # 环境变量模板
├── configs/
│   └── settings.yaml               # 业务口径阈值与双模运行配置
├── docker/
│   └── docker-compose.yml          # 16GB 内存友好的 HDFS + HMS + Spark 集群编排
├── dbt_project/                    # dbt-core 数仓建模、测试与审计配置
│   ├── dbt_project.yml
│   ├── profiles.yml
│   ├── models/
│   └── tests/
├── docs/
│   ├── archive/initial_draft.md    # 初始方案归档
│   ├── architecture.md             # 四层数仓、OPS 监控表与计算协同设计
│   └── rag_metrics_dictionary.md   # 供应链指标字典（Dify RAG 知识库）
├── scripts/
│   └── check_env.py                # 阶段 0 全栈环境自检脚本
├── src/lakepulse/                  # 核心 Python 包
│   ├── config.py                   # 双模式配置与 URI 解析
│   ├── spark/                      # SparkSession (JDK21 兼容) 与跨平台 Parquet I/O
│   ├── etl/                        # 阶段 1：数据接入、分布式清洗与审计监控
│   ├── analytics/                  # 阶段 2：ABC 分层与窗口特征工程
│   ├── ml/                         # 阶段 3：Baseline、TF Wide & Deep 与批量推理
│   └── api/                        # 阶段 4：FastAPI SQL 只读网关与 OpenAPI
└── tests/                          # pytest 自动化测试套件
```

---

## 六、快速开始（阶段 0 环境自检）

### 1. 创建独立 Python 3.11 + OpenJDK 17 环境（推荐 Conda）

> **注意**：`PySpark 3.5.x` 内置 Apache Arrow 12（含 `@pandas_udf`）与 Windows 版 `TensorFlow 2.x` 需运行在 **Python 3.11 + OpenJDK 17** 环境下；环境内安装 `openjdk=17` 可与宿主机全局 JDK 21 隔离共存。

```powershell
conda create -n lakepulse python=3.11 openjdk=17 -y
D:\miniconda\envs\lakepulse\python.exe -m pip install -e ".[dev]"
```

### 2. 执行自检、测试与流水线命令

| 操作 | Windows (PowerShell) | Linux / macOS / WSL |
| :--- | :--- | :--- |
| **全栈环境自检** | `.\make.ps1 check` | `make check` |
| **运行单元测试** | `.\make.ps1 test` | `make test` |
| **代码静态检查** | `.\make.ps1 lint` | `make lint` |
| **校验 Docker 集群配置** | `.\make.ps1 docker-config` | `make docker-config` |
| **阶段 1：ODS 分布式清洗** | `.\make.ps1 etl` | `make etl` |
| **阶段 1：dbt 建模与门禁** | `.\make.ps1 dbt` | `make dbt` |
| **阶段 1：一键全链路跑批** | `.\make.ps1 stage1` | `make stage1` |
| **阶段 2：经营诊断与特征工程** | `.\make.ps1 stage2` | `make stage2` |
| **阶段 3：模型训练与对比评估** | `.\make.ps1 train` | `make train` |
| **阶段 3：分布式批量补货推理** | `.\make.ps1 infer` | `make infer` |
| **阶段 3：一键训练+推理闭环** | `.\make.ps1 stage3` | `make stage3` |
| **阶段 4：启动 FastAPI 网关** | `.\make.ps1 serve` | `make serve` |
| **阶段 4：Agent 闭环演示** | `.\make.ps1 stage4` | `make stage4` |
| **端到端全链路一键跑通** | `.\make.ps1 all` | `make all` |

---

## 七、5 阶段实施路线图

| 阶段 | 核心任务 | 状态 |
| :--- | :--- | :--- |
| **阶段 0：立项重构与工程骨架** | 正式文档、双模配置、16GB Docker Compose、dbt 骨架、PySpark/TF/DuckDB 自检通过 | **已完成** |
| **阶段 1：数据清洗、数仓建模与监控** | M5 数据接入/抽样、PySpark 宽表转行表、dbt 四层建模与 `ops_pipeline_audit` 监控 | **已完成** |
| **阶段 2：SQL 经营分析与窗口特征宽表** | ABC 分层、节假日同环比、折扣分析、无数据穿越 Spark Window 特征宽表 | **已完成** |
| **阶段 3：Baseline、TF 训练与分布式推理** | Naive / LightGBM 对照、TensorFlow `Wide & Deep` 训练、`@pandas_udf` 批量补货计算 | **已完成** |
| **阶段 4：FastAPI 安全网关与 Dify Agent** | `sqlglot` 只读防火墙接口、指标 RAG 知识库与智能补货工作流闭环 | **已完成** |
