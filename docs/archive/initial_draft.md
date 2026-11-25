项目题目
LakePulse-AI：企业级湖仓一体智能经营与风控决策平台

核心定位
以电商与支付业务为背景，构建从底层分布式数据治理到深度学习预测、再到大模型自然语言决策的企业级闭环开源系统，同时支撑用户流失预警（增长线）与交易反欺诈识别（风控线）两大核心业务。

完整技术栈
分布式存储与计算： Hadoop 3.x (HDFS / YARN)、Hive Metastore、Apache Spark 3.5 (PySpark)、Parquet 列式存储

数仓建模与质量门禁： dbt-core、DuckDB

深度学习与分布式推理： TensorFlow 2.x (Keras Wide & Deep 流失模型 + Deep & Cross 风控模型)、PySpark @pandas_udf（Executor 端分布式批量推理）

数据服务安全网关： FastAPI（SQL 只读防火墙、自动 OpenAPI 文档）、Cloudflare Tunnel / ngrok（公网穿透）

AI 交互与知识库前台： Dify Cloud（多分支工作流 + 数仓指标 RAG 知识库）

部署与工程治理： Docker Compose（适配本地 16GB 内存的轻量伪分布式集群 + Local 双模式）、Makefile、AGENTS.md 规范
