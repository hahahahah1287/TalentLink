# TalentLink AI 系统测试报告

## 1. 测试概述

本报告记录 TalentLink AI 系统的三项评估结果：RAG 检索质量评估、全链路端到端评估（Contract Review + Research Agent）、Review Agent 评估。

测试环境：
- 模型：Qwen3.5-9B-Q5_K_M.gguf（本地 GGUF）
- Embedding：BAAI/bge-m3（CPU）
- Reranker：BAAI/bge-reranker-v2-m3（CPU）
- 检索：BM25 + FAISS 混合检索 → Cross-Encoder Rerank
- 知识库：《中华人民共和国劳动法》全文（树状切分，带元数据标注）

---

## 2. 测试框架与方法

### 2.1 RAG 检索质量评估

**脚本**：`tests/eval_ragas.py`

**方法**：自定义非 LLM 指标，不依赖第三方 AI 做 Judge。

**指标定义**：
- **Context Precision（上下文精确度）**：检索结果中包含了多少查询关键词。计算方式：`query 关键词 ∩ 检索结果关键词 / query 关键词总数`
- **Context Recall（上下文召回率）**：标准答案中的关键信息是否被检索到。计算方式：`ground_truth 关键词 ∩ 检索结果关键词 / ground_truth 关键词总数`

**分词方式**：jieba 中文分词 + 法条引用正则（`第X条/章/节/款`）+ 英文词 + 数字，过滤停用词。

**测试数据来源**：`tests/ragas_dataset.py`，100 条人工标注数据：
- 覆盖《劳动法》全部 13 章 107 条
- 合同审查场景 43 条（试用期、加班费、竞业限制、社保、解除合同等）
- 其他场景 57 条（法律咨询、条文解读、劳动争议等）
- 每条包含 `question`（用户问题）、`ground_truth`（标准答案）、`scene`（场景标签）

**检索链路**：HyDE 查询改写 → BM25+FAISS 混合检索（k=8）→ Cross-Encoder Rerank（top_k=5, threshold=0.3）

---

### 2.2 全链路端到端评估

**脚本**：`tests/eval_e2e.py`

**方法**：直接调用 LangGraph workflow，从输入到输出完整跑通，测量端到端指标。

**指标定义**：
- **Pipeline Completion Rate**：workflow 是否正常完成并产出 final_answer（长度 > 10 字符）
- **Tool Chain Coverage**：实际工具调用序列与预期工具链的覆盖率（集合交集/预期集合大小）
- **Review Status Accuracy**：Review Agent 的 approve/revise 判断与预期是否一致（仅 contract 路径）
- **Answer Relevance**：最终回答与 ground_truth 的 jieba 关键词重叠度（与 RAG Recall 同一算法）
- **Latency**：从 workflow 启动到返回结果的端到端耗时

**测试数据来源**：

Contract Review（10 条）：每条包含合同片段 + 法律审查问题 + 标准答案 + 预期工具链 `["legal_retrieval", "contract_chain", "review_agent"]` + 预期 review_status。

Research Agent（5 条）：每条包含法律问题 + 标准答案 + 预期工具链（含 `legal_search` / `web_search` + `synthesis_chain` + `review_agent`）。

**Contract Review 流程**：`retrieve → analyze → review`（Review Agent 发现引用问题时触发 re_synthesize 重试）

**Research Agent 流程**：`plan → execute（循环）→ synthesize → review`（Planner 自动选择工具）

---

### 2.3 Review Agent 评估

**脚本**：`tests/eval_review_agent.py`

**方法**：基于 LangSmith 框架，使用确定性指标（无 LLM Judge）。

**指标定义**：
- **Tool Call Accuracy**：是否调用了预期的审查工具（pii_check / citation_check / quality_check / add_disclaimer）
- **Review Status Accuracy**：approve/revise 判断是否正确
- **Completion Rate**：是否在限定步骤内正常完成

**测试数据来源**：10 条覆盖 approve（引用正确）、revise（引用不存在的法条）、PII 检测、质量检查、综合场景。

---

## 3. 测试结果

### 3.1 RAG 检索质量（100 条）

| 指标 | Overall | Contract (n=43) | Other (n=57) |
|---|---|---|---|
| Context Precision | **0.6452** | 0.6454 | 0.6451 |
| Context Recall | **0.7976** | 0.7415 | 0.8400 |

**分析**：
- Recall 0.80 表示标准答案中 80% 的关键信息被检索到，检索链路基本可用
- Precision 0.65 偏低，约 35% 的检索关键词不在查询中，说明检索结果有一定噪声
- Contract 场景 Recall（0.74）低于 Other（0.84），合同审查问题的表述方式与法条原文差异更大，HyDE 改写效果有限

---

### 3.2 Contract Review 全链路（10 条）

| 指标 | 分数 |
|---|---|
| Pipeline Completion Rate | **1.0000** |
| Tool Chain Coverage | **1.0000** |
| Review Status Accuracy | **1.0000** |
| Answer Relevance | **0.7654** |
| Avg Latency | **34.24s** |
| Errors | **0** |

逐条明细：

| # | 问题 | Relevance | Tool Cov | Review | Latency |
|---|---|---|---|---|---|
| 1 | 试用期约定是否合法 | 0.84 | 1.00 | approve | 35.8s |
| 2 | 加班费条款是否合法 | 0.84 | 1.00 | approve | 35.7s |
| 3 | 合同解除条款是否合法 | 0.68 | 1.00 | approve | 33.8s |
| 4 | 竞业限制条款是否合理 | 0.61 | 1.00 | approve | 33.8s |
| 5 | 社会保险条款是否合规 | 0.94 | 1.00 | approve | 33.6s |
| 6 | 年假约定是否合法 | 0.69 | 1.00 | approve | 33.4s |
| 7 | 工资支付条款是否合规 | 0.47 | 1.00 | approve | 34.3s |
| 8 | 违约金条款是否合法 | 0.93 | 1.00 | approve | 34.3s |
| 9 | 工伤条款是否合法 | 0.75 | 1.00 | approve | 33.6s |
| 10 | 培训费用条款是否合法 | 0.89 | 1.00 | approve | 33.9s |

**分析**：
- Completion 和 Tool Chain 均为满分，pipeline 稳定可靠
- Review Status 全部正确，Review Agent 未出现误判
- Relevance 0.77 与 RAG Recall 0.80 接近，瓶颈在检索质量而非生成质量
- 低分用例（工资支付 0.47、竞业限制 0.61）原因：回答使用了不同于 ground_truth 的表述方式，关键词重叠度低但内容方向正确

---

### 3.3 Research Agent 全链路（5 条）

| 指标 | 分数 |
|---|---|
| Pipeline Completion Rate | **1.0000** |
| Tool Chain Coverage | **0.9500** |
| Answer Relevance | **0.7767** |
| Avg Latency | **30.27s** |
| Errors | **0** |

逐条明细：

| # | 问题 | Planner 选择 | Relevance | Tool Cov | Latency |
|---|---|---|---|---|---|
| 1 | 劳动法第三十八条规定内容 | legal_search | 1.00 | 1.00 | 23.3s |
| 2 | 2024年最低工资标准 | web_search | 0.58 | 0.75 | 31.5s |
| 3 | 无固定期限劳动合同条件 | legal_search | 0.94 | 1.00 | 34.2s |
| 4 | N+1补偿怎么规定 | legal_search | 0.89 | 1.00 | 37.0s |
| 5 | 劳动仲裁时效 | legal_search | 0.47 | 1.00 | 25.3s |

**分析**：
- Planner 工具选择基本正确：法律问题选 legal_search，需要最新信息的选 web_search
- 第 2 条 Tool Cov 0.75：Planner 只选了 web_search，预期是 legal_search + web_search 都选，但逻辑上仅用 web_search 查最新标准也合理
- Latency（30.3s）比 Contract（34.2s）快约 4s，因为 Research 路径的检索不走 HyDE

---

## 4. 两条链路对比

| 指标 | Contract Review | Research Agent |
|---|---|---|
| Completion Rate | 1.00 | 1.00 |
| Answer Relevance | 0.77 | 0.78 |
| Tool Chain Coverage | 1.00 | 0.95 |
| Review Status Accuracy | 1.00 | — |
| Avg Latency | 34.2s | 30.3s |

---

## 5. 已知问题与优化方向

| 问题 | 影响 | 优化方向 |
|---|---|---|
| RAG Precision 偏低（0.65） | 检索结果含噪声，影响生成质量 | 调整 score_threshold、优化 jieba 自定义词典 |
| Answer Relevance 上限约 0.8 | 关键词匹配无法衡量语义相似度 | 引入 embedding 相似度作为补充指标 |
| Contract 场景 Recall 低于 Other | 合同审查 query 与法条表述差异大 | 优化 HyDE prompt、增加合同领域训练数据 |
| HyDE 效果不明显 | Precision 不变，Recall 微降 | 测试不同 HyDE prompt 模板 |
| Latency 30-35s 偏高 | 9B 模型推理 + CPU Rerank | Reranker 移至 GPU、减小 n_ctx |

---

## 6. 测试数据集统计

| 数据集 | 条数 | 用途 | 位置 |
|---|---|---|---|
| RAGAS Dataset | 100 条（contract 43, other 57） | RAG 检索质量评估 | `tests/ragas_dataset.py` |
| Contract E2E Dataset | 10 条 | Contract Review 全链路评估 | `tests/eval_e2e.py` 内嵌 |
| Research E2E Dataset | 5 条 | Research Agent 全链路评估 | `tests/eval_e2e.py` 内嵌 |
| Review Agent Dataset | 10 条 | Review Agent 子 agent 评估 | `tests/eval_review_agent.py` 内嵌 |
