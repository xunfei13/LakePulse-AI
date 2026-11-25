# LakePulse-AI 供应链经营与补货指标字典（RAG 知识库）

> 本文档定义系统所有对外指标的标准业务口径，供研发、业务运营及 **Dify AI Agent RAG 知识库** 统一引用，消除“指标口径幻觉”。

---

## 1. 商品价值分层指标

### ABC 商品等级（`abc_class`）
按统计期内同一仓库（或全域）各商品累计有效销售额降序排列，依据**累计销售额占比**划分：
- **A 类核心商品**：累计销售额占比 $\le 70\%$ 的头部商品（贡献主要营收，断货损失最高）。
- **B 类主力商品**：累计销售额占比在 $(70\%, 90\%]$ 区间的腰部商品。
- **C 类长尾商品**：累计销售额占比在 $(90\%, 100\%]$ 区间的长尾慢销商品。

---

## 2. 销售与价格口径

### 有效日销售额（`revenue`）
- **计算公式**：$\text{revenue} = \text{sales} \times \text{sell\_price}$
- **清洗规则**：仅纳入已在售商品（`sell_price IS NOT NULL` 且 `sell_price > 0`、`sales >= 0`）；未上架前周期记录不计入分母均值。

### 价格折扣率（`price_discount_rate`）
- **计算公式**：$\text{price\_discount\_rate} = \frac{\text{price\_ma\_28} - \text{sell\_price}}{\text{price\_ma\_28}}$
- **业务解读**：正值代表当前售价低于近 28 天均价（处于降价/促销状态），例如 `0.15` 表示降价 15%；负值或 0 表示原价或涨价。

---

## 3. 预测与补货决策口径

### 未来 7 天预测销量（`pred_next_7d_sales`）
- 由 **TensorFlow Wide & Deep** 回归模型基于截止预测基准日 $T$ 的历史滑窗销量、价格折扣与未来 7 天已知日历排期（周末/节假日/SNAP 消费券）计算得出，并截断负值（$\ge 0$）。

### 安全缓冲天数规则（`safety_buffer_days`）与安全缓冲库存（`safety_stock_qty`）
为对冲需求波动并防止头部畅销品缺货，按商品价值等级设定差异化缓冲天数：
- **A 类商品**：缓冲 **3.0 天**
- **B 类商品**：缓冲 **2.0 天**
- **C 类商品**：缓冲 **1.0 天**

$$\text{safety\_stock\_qty} = \frac{\text{pred\_next\_7d\_sales}}{7.0} \times \text{safety\_buffer\_days}$$

### 建议补货量（`recommended_order_qty`）

$$\text{recommended\_order\_qty} = \max\Bigl(0,\; \lceil \text{pred\_next\_7d\_sales} + \text{safety\_stock\_qty} - \text{current\_available\_stock} \rceil\Bigr)$$

### 断货风险等级（`stockout_risk_level`）
结合 **当前可用库存对未来 7 天预测需求的覆盖天数**（`coverage_days = current_available_stock / max(pred_next_7d_sales / 7.0, 0.01)`）判定：
- **HIGH（高风险）**：`coverage_days < 3.0`（或建议补货量 $> 0$ 且当前库存不足以支撑未来 3 天预测销量）
- **MEDIUM（中风险）**：$3.0 \le \text{coverage\_days} < 7.0$
- **LOW（低风险）**：$\text{coverage\_days} \ge 7.0$

---

## 4. 数据质量与跑批监控口径

### 跑批健康状态（`ops_pipeline_audit.status`）
- **SUCCESS**：任务正常完成，主键唯一性校验、非负校验全部通过，输出行数波动在正常阈值范围内。
- **WARN**：任务产出可用，但存在轻度异常信号（如单表空值率偏高或行数较上批次波动较大）。
- **FAILED**：质量门禁阻断或运行异常，下游补货计算暂停执行。
