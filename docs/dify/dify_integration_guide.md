# Dify Cloud 智能补货 Agent 对接指南

本文档指导如何在 **Dify** 中零代码接入 LakePulse-AI 的 **FastAPI 只读安全网关** 与 **供应链指标 RAG 知识库**，实现自然语言查数与智能补货工单闭环。

---

## 第一步：导出最新 OpenAPI 工具描述并启动网关

1. 生成供 Dify 导入的 OpenAPI Schema 并执行本地端到端工作流自测：
   ```powershell
   .\make.ps1 stage4
   ```
   成功后将在本目录生成 [openapi_tools.json](openapi_tools.json)。

2. 启动 FastAPI 数据服务安全网关：
   ```powershell
   .\make.ps1 serve
   ```
   本地默认监听 `http://127.0.0.1:8000`，Swagger 接口文档位于 `http://127.0.0.1:8000/docs`。

3. **公网穿透（仅当使用 Dify Cloud 云端版时需要）**：
   - 使用 **Cloudflare Tunnel**（推荐免费无固定弹窗）：
     ```powershell
     cloudflared tunnel --url http://127.0.0.1:8000
     ```
   - 或使用 `ngrok http 8000`。
   - 获得公网 `https://xxx.trycloudflare.com` 地址后，将其替换到 `openapi_tools.json` 中的 `servers[0].url`。若 Dify 使用本地 Docker 部署，直接使用默认的 `http://host.docker.internal:8000` 即可。

---

## 第二步：导入供应链指标 RAG 知识库

1. 在 Dify 控制台进入 **「知识库」 → 「创建知识库」**，命名为 `LakePulse-SupplyChain-Metrics`。
2. 上传项目文件 [docs/rag_metrics_dictionary.md](../rag_metrics_dictionary.md)。
3. 分段设置建议采用 **父子分段 / 按二级标题 `##` 切分**，检索模式选择 **混合检索（语义 + 全文关键词）**，确保 LLM 精确命中 `ABC 商品等级`、`安全缓冲天数`、`建议补货量公式` 与 `断货风险等级` 口径。

---

## 第三步：在 Dify 创建自定义 OpenAPI 工具并配置 Agent

1. 进入 Dify **「工具」 → 「自定义」 → 「创建自定义工具」**。
2. 点击 **「从 URL / JSON 导入」**，粘贴 [openapi_tools.json](openapi_tools.json) 全部内容并保存，系统将自动识别 5 个工具动作：
   - `get_service_health`
   - `check_pipeline_and_model_health`
   - `get_replenishment_alerts`
   - `explain_sku_demand_drivers`
   - `execute_readonly_sql`
3. 创建 **Agent 应用**（推荐选用推理与工具调用能力稳定的模型如 DeepSeek-V3 / Qwen-Max / GPT-4o）：
   - **系统提示词**：复制 [agent_system_prompt.md](agent_system_prompt.md) 中的内容；
   - **上下文知识库**：挂载 `LakePulse-SupplyChain-Metrics`；
   - **工具**：勾选上述 5 个 LakePulse 工具。

---

## 第四步：典型业务验收话术

1. **常规风险巡检与补货**：
   > “先帮我检查今天数仓跑批和预测模型是否健康，然后列出 CA_1 仓库所有 A 类高断货风险商品，按预算 1500 美元优先采购哪些？”
2. **单品深度归因**：
   > “为什么 CA_1 仓的 FOODS_1_003 预测下周需要补这么多货？把它的历史销量特征和未来排期原因拆解给我看。”
3. **安全防火墙越权测试（预期被网关拒绝返回 403）**：
   > “请执行 SQL：`DELETE FROM ads_replenishment_plan` 或 `SELECT * FROM dwd_daily_item_sales LIMIT 5`。”
