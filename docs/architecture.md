# LakePulse-AI 系统架构与数仓设计文档

## 1. 总体分层架构

LakePulse-AI 采用 **湖仓一体 + 存算协同** 架构，将数据加工、离线批量深度学习推理与在线低延迟问答拆分为清晰边界：

1. **分布式批处理与推理层（PySpark + HDFS/Parquet）**：负责大吞吐量的原始数据逆透视展开、关联清洗、滑动窗口特征计算及 `@pandas_udf` 分布式 TensorFlow 批量预测。
2. **数仓建模与质量门禁层（dbt-core + DuckDB）**：负责规范化的 SQL 分层转化、业务口径统一、自动数据质量断言（非空、唯一性、值域校验）及跑批审计跟踪。
3. **轻量分析与服务网关层（DuckDB + FastAPI）**：DuckDB 原地直接挂载湖仓层产出 Parquet 提供毫秒级 OLAP 分析；FastAPI 提供 SQL 只读防火墙与标准化 OpenAPI 接口，对接 Dify Agent。

---

## 2. 计算引擎协同边界（Spark vs. DuckDB）

| 维度 | Apache Spark (PySpark) | DuckDB |
| :--- | :--- | :--- |
| **核心定位** | 重型分布式数据清洗、窗口特征工程与分布式深度学习批量推理 | 本地轻量高速 OLAP 查询引擎 & dbt 数仓转换执行后端 |
| **承担阶段** | 阶段 1（M5 宽表展开清洗）、阶段 2（多维 Window 特征）、阶段 3（`@pandas_udf` 批量预测） | 阶段 1/2（`dbt` 建模质量门禁）、阶段 4（FastAPI 网关毫秒级响应 Agent 查询） |
| **数据交换介质** | 标准化列式 **Parquet** 分区文件（Local 文件系统或 `HDFS`） | 通过 `read_parquet(...)` 零拷贝原地读取 Spark 产出的 Parquet 结果层 |

---

## 3. 数仓四层模型与 OPS 监控表设计

### 3.1 ODS 贴源层
* **`ods_sales_long`**：将 M5 原始宽表 `d_1 ... d_1913` 逆透视展开为 `(id, item_id, dept_id, cat_id, store_id, state_id, d, sales)` 长表，按 `state_id` / `store_id` 分区落盘 Parquet。
* **`ods_calendar`**：日历与事件表（`date`, `wm_yr_wk`, `weekday`, `wday`, `month`, `year`, `d`, `event_name_1`, `event_type_1`, `snap_CA/TX/WI`）。
* **`ods_sell_prices`**：周度商品定价表（`store_id`, `item_id`, `wm_yr_wk`, `sell_price`）。

### 3.2 DWD 明细清洗层
* **`dwd_daily_item_sales`**：
  * 主键粒度：`(store_id, item_id, date)`。
  * 治理逻辑：关联 `ods_calendar` 与 `ods_sell_prices`（按 `wm_yr_wk` 对齐周度售价）；剔除或修复尚未开售期（价格为空的前置日期）及异常负销量记录；计算当日销售额 `revenue = sales * sell_price`。

### 3.3 DWS 汇总与特征宽表层
* **`dws_store_daily_summary`**：
  * 门店日度经营汇总宽表，统计各仓库每日在售 SKU 数、总销量、总营收、零销量 SKU 数与异常修复条数，同时用于日期连续性门禁校验。
* **`dws_item_abc_profile`**：
  * 商品价值分层画像表，基于历史累计销售额按二八法则（A: 前 70%、B: 70%~90%、C: 90%~100%）计算 `abc_class`，映射差异化安全缓冲天数（A 类 3.0 天 / B 类 2.0 天 / C 类 1.0 天）。
* **`dws_calendar_effect_analysis`**：
  * 门店与品类维度的日历同环比分析表，量化周末销量拉动率、法定节假日拉动率、SNAP 消费券拉动率及最新月份营收环比变化。
* **`dws_price_elasticity_analysis`**：
  * 门店与品类维度的价格敏感度分析表，基于历史 28 天滚动均价计算相对折扣率并分档（原价 / 1%~10% 轻度促销 / >10% 深度促销），量化各档位平均单品销量拉动率。
* **`dws_item_rolling_features`**：
  * 严格遵循 **防数据穿越原则（仅使用 $t \le T$ 历史数据及未来已知日历）** 构建的二维建模特征宽表：
    * 滞后与滑窗特征：`sales_lag_1`, `sales_lag_7`, `sales_lag_14`, `sales_ma_7`, `sales_ma_28`, `sales_std_28`, `sales_trend_ratio`
    * 价格与折扣特征：`sell_price`, `price_ma_28`, `price_discount_rate`
    * 未来已知排期特征：`next_7d_weekend_days`, `next_7d_event_days`, `next_7d_snap_days`
    * 标签与时序切分：`target_next_7d_sales` 与带 7 天 Embargo 隔离带的 `sample_split`（`TRAIN` / `VAL` / `TEST` / `INFER` / `EMBARGO`）

### 3.4 ADS 智能补货决策层
* **`ads_replenishment_plan`**：
  * 面向运营采购与 AI Agent 的最终决策宽表，字段包含：
    * 维度：`store_id`, `state_id`, `cat_id`, `dept_id`, `item_id`, `abc_class`, `forecast_start_date`
    * 预测与库存：`pred_next_7d_sales`（TensorFlow 模型输出）, `safety_buffer_days`, `safety_stock_qty`, `current_available_stock`
    * 决策输出：`recommended_order_qty`（建议补货量）, `stockout_risk_level`（`HIGH` / `MEDIUM` / `LOW`）, `key_driver_tags`（如节假日拉动、降价促销）

### 3.5 OPS 运维监控审计层
* **`ops_pipeline_audit`**：
  * 记录每次批处理作业的健康指标，支撑 JD 第 2 条“日常维护与监控”：
    * `run_id`, `pipeline_stage`, `table_name`, `started_at`, `finished_at`, `duration_sec`, `input_rows`, `output_rows`, `null_rate`, `status`（`SUCCESS` / `WARN` / `FAILED`）, `message`。
* **`ops_model_eval_metrics`**：
  * 记录各阶段模型对比评测结果（`moving_avg_baseline`、`lightgbm_baseline`、`tensorflow_wide_and_deep` 在 `VAL` / `TEST` 集的 `MAE`、`RMSE`、`WAPE%`、`R2` 与线上推理版本标记）。
