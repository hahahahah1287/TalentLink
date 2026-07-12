# TalentLink — 劳动法务可信 RAG + 受控多专家 Agent 系统

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/LangGraph-0.2.x-green.svg" alt="LangGraph">
  <img src="https://img.shields.io/badge/LLM-Qwen3.5-9B-purple.svg" alt="Qwen">
  <img src="https://img.shields.io/badge/Reranker-bge--v2--m3-orange.svg" alt="Reranker">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

> 面向劳动法务场景的可信 RAG + 受控多专家 Agent 系统。通过**确定性 workflow、结构化专家输出、推理服务化和分层评测**降低 LLM 幻觉与工程不稳定性。知识库为《中华人民共和国劳动法》107 条；定位是"有证据、可追溯、能审计"的法务助手，而非通用聊天机器人。
>
> **核心叙事：把 LLM 关进笼子**——法律判定交给确定性 skill 与规则，LLM 只做语言合成与查询改写。

---

## 架构

系统收敛为**一条确定性法务图**：`retrieve → run_skills → generate → guard`。没有 Planner、没有 ReAct、没有 LLM 自由调度，编排权完全在确定性状态机手里。

```
                         ┌─────────────────────────────────────────────┐
                         │            WorkflowService                   │
                         │  intent_router(deterministic-router-v3)     │
                         │  精确 checkpoint 缓存（WorkflowCheckpoint） │
                         │  local/server 可切换 LLM backend            │
                         └──────────────────────┬──────────────────────┘
                                                │
                                  scene = legal | chat
                                                │
                              ┌─────────────────▼─────────────────┐
                              │   build_legal_graph (StateGraph)  │
                              │   set_entry_point("retrieve") 硬编码│
                              └─────────────────┬─────────────────┘
                                                │
                  ┌─────────────────────────────▼─────────────────────────────┐
                  │  1. retrieve  (入口，永远先执行，不可被路由绕过)            │
                  │     HyDE 改写 → BM25+FAISS 混合检索 → bge-reranker 精排   │
                  │     产出 evidence_items / law_context                    │
                  └─────────────────────────────┬─────────────────────────────┘
                                                │
                  ┌─────────────────────────────▼─────────────────────────────┐
                  │  2. run_skills  (Bounded Parallel Specialist Agents)      │
                  │     asyncio.gather 并行调度，每个 specialist 固定职责：    │
                  │     ┌────────────┐ ┌────────────┐ ┌────────────┐         │
                  │     │ risk_clause│ │ compliance │ │ term expl.│ ...     │
                  │     └─────┬──────┘ └─────┬──────┘ └─────┬──────┘         │
                  │           │  normalize → validate → clean               │
                  │           └──────────────┬──────────────┘               │
                  │     skill-result-v1 统一 schema + 纠错（specialist_     │
                  │     corrections / specialist_reports）                  │
                  │     local 模式 LLM-bound skill 加 asyncio.Lock 串行     │
                  └─────────────────────────────┬─────────────────────────────┘
                                                │
                  ┌─────────────────────────────▼─────────────────────────────┐
                  │  3. generate  (LLM 只做语言合成，不做法律判断)            │
                  │     prompt 变量仅 {law}{contract}{skill_findings}{question}│
                  │     只消费清洗后的 skill_outputs + law_context           │
                  │     retry 时追加"只引用证据、不要编造条文号"约束         │
                  └─────────────────────────────┬─────────────────────────────┘
                                                │
                  ┌─────────────────────────────▼─────────────────────────────┐
                  │  4. guard  (输出端校验)                                  │
                  │     CitationGuard → 引用是否在 evidence 白名单           │
                  │     ┌── 有问题 & guard_retry<1 ──→ "revise" 回 generate  │
                  │     └── 无问题 → GuardrailsPipeline.run                │
                  │              (PII → 质量 → 引用 → 免责声明) → END        │
                  └──────────────────────────────────────────────────────────┘
```

### 各节点职责

| 节点 | 职责 | 关键产物 |
|------|------|----------|
| **retrieve** | HyDE 改写 + BM25/FAISS 混合检索 + bge-reranker 精排，作为图入口永远先执行 | `evidence_items`、`law_context` |
| **run_skills** | Bounded Parallel Specialist Agents 并行执行；输出经 `normalize → validate → clean` 清洗 | `skill_outputs`、`specialist_reports`、`specialist_corrections` |
| **generate** | LLM 只做语言合成，prompt 仅含已验证的 finding + 法条原文，不做法律判断 | `draft_answer` / `final_answer` |
| **guard** | 输出端校验：引用是否在证据白名单、PII 脱敏、免责声明；发现问题可 `revise` 回 generate 一次 | `guard_issues`、`guard_retry` |

### 确定性路由

`utils/intent_router.py` 是纯函数加权关键词评分（`ROUTE_VERSION="deterministic-router-v3"`，零 LLM），按 query 决定路由哪些 specialist，结果写进 `route_skills`。

---

## Bounded Parallel Specialist Agents

specialist 不是普通 tool：每个有固定职责、固定输入 `fn(query, law_context="")`、统一 `skill-result-v1` 输出 schema，并经校验+纠错。

| Specialist | 职责 | 判定方式 |
|------------|------|----------|
| `risk_clause_detector` | 风险条款识别 | LLM 抽取 ContractFacts + 规则引擎 `_rule_probation` 等纯函数判定 |
| `compliance_check` | 合规检查 | 外部决策表 `data/compliance_rules.json` + 通用算子 `_apply_op` |
| `legal_term_explainer` | 法律术语解释 | LLM |
| `statute_checker` | 时效/法条适用计算 | 确定性计算 |
| `case_retriever` | 案例检索 | bge + element_boost + mmr 检索 |

### 输出校验与纠错（输入端校验）

specialist 输出被视为**不可信输入**，进 generate 前必经三阶段（`utils/skill_result.py`）：

1. **normalize** — 把任意输出归一为 `skill-result-v1` envelope（已是 v1 的补缺字段+修 skill_name；legacy dict 走 `make_skill_result(..., legacy=parsed)`；纯文本兜底）
2. **validate** — 检 schema_version / status / facts / metrics / provenance / findings / evidence
3. **clean** — 剔除空 finding（`empty_finding`）、非 dict finding（`finding_not_object`）、非法 evidence

所有修正/剔除动作写进 `specialist_corrections`，per-specialist 报告写进 `specialist_reports[name]`。**原则：能确定性修复的才修（补 schema、改 skill_name）；不能的剔除，绝不让 LLM 凭空补法律依据。**

### guard（输出端校验）

`utils/guardrails.py` 的 `GuardrailsPipeline` 默认顺序 `[PIIGuard, QualityGuard, CitationGuard, DisclaimerGuard]`。`CitationGuard` 从答案提取 `《法律名》第X条` 与 `evidence_law_article_pairs` 白名单比对，未命中触发一次 `revise` 回 generate 重合成。

> **输入端校验 vs 输出端校验**：`validate_and_clean` 管"专家结论可不可信"（进 generate 前），`guardrails` 管"最终回答安不安全"（出 generate 后）。

---

## LLM 推理服务化与可切换后端

排查并解决了本地 9B GGUF 在 `llama-cpp-python` 进程内多线程并发 decode 导致的 `native abort 134`：多个 LLM-bound specialist 共享同一个 llama.cpp context 并发 decode，Python 层看不到普通异常，进程直接挂。

| 项 | local 模式 | server 模式 |
|----|-----------|-------------|
| 客户端 | `ChatLlamaCpp`（直连 GGUF，进程内） | `ChatOpenAI`（OpenAI-compatible HTTP client） |
| 模型加载 | 业务进程加载一份 | 仅 `llama_cpp.server` 进程加载一份，业务进程不加载 |
| 并行 | LLM-bound specialist 加 `asyncio.Lock` 串行 | 无锁，`asyncio.gather` 真并行发 HTTP |
| 适用 | 离线/简单 demo | 生产/E2E 评测，更稳定 |

统一封装在 `utils/llm_factory.py`：`create_chat_llm` 二选一（互斥，不会加载两份），`llm_supports_parallel_requests()` 仅 server 返回 True，据此决定哪些 specialist 标 `llm_bound`。环境变量 `LLM_BACKEND` / `LLM_SERVER_BASE_URL` / `LLM_SERVER_MODEL` / `LLM_SERVER_API_KEY` 覆盖配置。切换 Ollama/vLLM/云模型只改三个环境变量，业务层不动。

---

## 三阶段检索管线

| 阶段 | 技术 | 作用 |
|------|------|------|
| 查询改写 | HyDE（`utils/query_rewriter.py`） | 口语化查询对齐到法言法语，评测里可用 `--no-hyde` 关闭做 baseline |
| 混合检索 | BM25 + FAISS Ensemble（`utils/retrieval_service.py`） | 关键词精确匹配 + 语义向量检索，互补 |
| 精排 | bge-reranker-v2-m3 Cross-Encoder（`utils/reranker.py`） | 深度语义打分，`score_threshold=0.3` 过滤低质证据 |

检索产出 `evidence_items`（带 `evidence_id`/`article_id`/`canonical_citation`/`text_hash`），贯穿 retrieve→generate→guard 作为可追溯引用根。

### 法律文档结构化处理

- **结构化切分**：`utils/legal_parser.py` 按 篇→章→节→条 切分，每条 `Document` 自带 `article`/`article_id`/`canonical_citation`/`text_hash` 等基础 metadata（不依赖 LLM）
- **元数据标注**：`utils/metadata_annotator.py` 用 LLM 补 `summary`/`keywords`/`applicable_scenario`/`legal_effect`（可选）
- **索引构建**：`build_index.py` 支持 `--skip-metadata` / `BUILD_INDEX_SKIP_METADATA` 快速路径（跳过 9B 加载，直接 bge-m3 向量化 + FAISS build）；二者是 **OR 关系**（任一为真即跳过）

---

## 证据链与审计字段

`utils/state.py` 的 `AppState` 不只存 `final_answer`，而是贯穿结构化审计字段，让系统能解释自己怎么得出结论：

| 字段 | 记录内容 |
|------|----------|
| `evidence_items` | article-level 证据（法条原文、`canonical_citation`、`text_hash`） |
| `tool_history` | 有序执行日志（route / retrieve / specialist_{name} / generate / guard_revise） |
| `skill_outputs` | 清洗后的 skill-result-v1，generate 只消费它 |
| `specialist_reports` | per-specialist 的 status / corrections / dropped_findings |
| `specialist_corrections` | 所有修正/剔除动作（`set_schema_version` / `drop_finding` 等） |
| `guard_issues` | 输出校验问题（引用未命中证据等） |

这些字段默认不进用户 prompt（generate 只吃 `skill_outputs`+`law_context`），只进审计/评测/RAGAS contexts 构造，不污染答案。

---

## 分层评测体系

| 层 | 脚本 | 回答的问题 | 当前结果 |
|----|------|-----------|----------|
| 检索 | `tests/eval_legal_retrieval.py` | 正确法条召回了没 | 20 条，hit@5/recall@5=0.90，MRR=0.7417，NDCG@5=0.7827 |
| E2E | `tests/eval_e2e.py` | 链路跑通没、工具挂对没 | 3 条 smoke，completion rate=100%，tool-chain coverage=93.33% |
| 专家 | `eval_e2e.py` 内 specialist metrics | 专家执行了没、有没有纠错 | specialist coverage=100%，correction=5，error=0 |
| RAGAS | `tests/eval_ragas.py` | 答案是否忠于证据、是否覆盖 reference | 3 条 smoke，faithfulness=0.6170，context_recall=0.6667 |

> **诚实边界**：E2E 和 RAGAS 都只有 3 条 smoke，只能证明链路、不能证明质量。正式结论需扩到 30-50 条分场景样本。详见 `docs/resume_talking_points.md`。

### RAGAS 评测输入优化

RAGAS 的 `retrieved_contexts` 必须与系统真实证据链一致。`tests/eval_ragas.py:_contexts_from_row` 把 `evidence_items` + `law_context` + `contract_text` + `skill_outputs`（经 `render_skill_result_for_prompt` 渲染）合并为 contexts。优化后单样本 `context_recall` 从 0.50→1.00、`faithfulness` 从 0.25→0.67。

防假报告：`evaluate(..., raise_exceptions=True)` 失败直接抛异常，summary 聚合时逐行过滤 NaN。Gemini 限速用 `threading.Lock` + `min_interval_s`（默认 13s），`--offset`/`--limit` 支持分批续跑。

---

## 缓存

`utils/checkpoint.py` 的 `WorkflowCheckpoint` 用 `sha256(query + contract_hash + route_skills + corpus_fingerprint + workflow_version)` 做**精确指纹**缓存。法务场景一字之差结论不同，不用语义相似缓存——避免"试用期 6 个月"与"试用期 6 个月合法吗"串答案。只缓存首轮（`not has_history`），Redis 不可用时自动降级无缓存。

> 注：`utils/semantic_cache.py`（余弦相似 0.93）是 legacy `services.py` 的遗留，当前 `WorkflowService` 未使用。

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 工作流编排 | LangGraph StateGraph（确定性 DAG） |
| LLM | Qwen3.5-9B（GGUF IQ4_XS，4.81GB） |
| 推理后端 | llama-cpp-python（local）/ llama_cpp.server（server，OpenAI-compatible） |
| Embedding | BAAI/bge-m3 |
| Reranker | BAAI/bge-reranker-v2-m3 |
| 向量数据库 | FAISS |
| 关键词检索 | BM25 |
| Web 框架 | FastAPI（SSE 流式） |
| 数据库 | MySQL + Redis |
| 评测 | RAGAS + article-level deterministic + E2E + specialist metrics |

---

## 快速开始

### 环境要求

- Python 3.13+
- MySQL 8.0+
- Redis 7.0+（可选，用于缓存）
- CUDA（可选，GPU 加速）

### 安装

```bash
git clone https://github.com/your-username/talentlink.git
cd talentlink
pip install -r requirements.txt

# 下载 GGUF 模型放在项目根目录
# Qwen3.5-9B-IQ4_XS.gguf (4.81GB)
```

### 构建索引

```bash
# 快速路径：跳过 9B metadata 标注，直接 bge-m3 向量化 + FAISS build
python build_index.py --skip-metadata
# 或 export BUILD_INDEX_SKIP_METADATA=1

# 精标路径：加载 9B 逐条标注 summary/keywords/applicable_scenario/legal_effect
python build_index.py
```

### 启动 LLM 后端（可选 server 模式）

```bash
# server 模式：单进程加载一份 GGUF，业务进程走 HTTP
export LLM_BACKEND=server
# 启动 llama_cpp.server（监听 :8000，OpenAI-compatible /v1）

# local 模式（默认）：业务进程直连 GGUF
export LLM_BACKEND=local
```

### 启动服务

```bash
python main.py
# 访问 http://localhost:8000/docs
```

### API 接口

```python
import requests

# 法务问答 / 合同审查（统一入口，合同审查由后端按内容自动判定）
response = requests.post(
    "http://localhost:8000/chat/stream",
    json={
        "user_id": "user_123",
        "query": "这份合同的试用期约定是否合法？",
        "scene": "legal",                      # "legal" | "chat"
        "contract_text": "试用期为6个月，劳动合同期限为1年。"
    },
    stream=True
)
```

主要端点：`POST /chat/stream`、`GET /history/{user_id}`、`GET /health`、`GET /metrics`、`POST /cache/invalidate`。

---

## 项目结构

```
talentlink/
├── main.py                      # FastAPI 入口（/chat/stream 等）
├── workflow_service.py          # WorkflowService（编排服务层、缓存、llm 接线）
├── build_index.py               # 索引构建（--skip-metadata 快速路径）
├── memory.py                    # 会话管理 (MySQL/Redis/摘要)
├── services.py                  # 旧版 UnifiedAgentService（含 legacy 语义缓存）
├── config/
│   └── __init__.py              # 集中配置 (LLMConfig.backend 等 dataclass)
├── workflows/
│   ├── legal_graph.py           # 确定性法务图 retrieve→run_skills→generate→guard ★
│   ├── contract_review.py       # （legacy）合同审查 workflow
│   ├── research_agent.py        # （legacy）研究型 workflow
│   ├── review_agent.py          # （legacy）Review Agent
│   └── shared_nodes.py          # 共享节点
├── skills/
│   ├── base.py                  # Skill 基类（create_skill_fn/create_skill_tool）
│   ├── registry.py              # SkillRegistry（不再服务 LLM Planner）
│   ├── risk_clause_detector.py # 风险条款（规则引擎）
│   ├── compliance_check.py      # 合规检查（决策表 data/compliance_rules.json）
│   ├── legal_term_explainer.py  # 术语解释
│   ├── statute_checker.py       # 时效/法条适用
│   ├── case_retriever.py        # 案例检索
│   └── web_search.py            # 外部检索
├── utils/
│   ├── retrieval_service.py     # 混合检索 + Rerank + 证据产出
│   ├── reranker.py              # bge-reranker-v2-m3 Cross-Encoder
│   ├── query_rewriter.py        # HyDE
│   ├── intent_router.py         # 确定性路由（deterministic-router-v3）
│   ├── llm_factory.py           # local/server 可切换 LLM 后端 ★
│   ├── skill_result.py          # skill-result-v1 envelope + normalize/validate/clean
│   ├── evidence.py              # EvidenceItem / 证据链
│   ├── guardrails.py            # GuardrailsPipeline（PII/Citation/Disclaimer）
│   ├── checkpoint.py            # WorkflowCheckpoint（精确指纹缓存）
│   ├── semantic_cache.py        # （legacy）语义缓存
│   ├── legal_parser.py          # 法律文档结构化切分
│   ├── legal_corpus.py          # 语料 + corpus_fingerprint
│   ├── metadata_annotator.py    # LLM 元数据标注
│   ├── embeddings.py            # bge-m3 封装
│   ├── state.py                 # AppState（审计字段定义）
│   ├── tool_call_parser.py      # ToolCallParserLLM（XML tool call 解析）
│   ├── circuit_breaker.py       # 熔断器
│   ├── security.py              # 安全检测
│   └── tools/
│       ├── contract_tools.py    # 合同/法务 LLM Chain（prompt | llm | StrOutputParser）
│       ├── guard_tools.py       # Guard → Tool 封装
│       ├── common_tools.py
│       └── job_tools.py
└── tests/
    ├── eval_legal_retrieval.py  # 检索评测（hit/recall/MRR/NDCG）
    ├── eval_e2e.py              # E2E + specialist metrics
    ├── eval_ragas.py            # RAGAS 评测（contexts 优化）
    ├── eval_llm_judge.py        # LLM Judge
    ├── eval_review_agent.py     # （legacy）Review Agent 评估
    ├── ragas_dataset.py         # 评测数据集（100 条）
    ├── _smoke_graph.py          # 图结构 + correction 烟雾测试
    ├── test_legal_parser.py
    ├── test_skills.py
    └── reports/                 # 评测报告 JSON
```

---

## 评测结果

详见 `tests/TEST_REPORT.md` 与 `tests/reports/`。

### 检索质量（20 条法条级评测，hyde=false baseline）

| 指标 | 值 |
|---|---|
| hit@1 | 0.60 |
| hit@5 | 0.90 |
| recall@5 | 0.90 |
| precision@5 | 0.18 |
| MRR | 0.7417 |
| NDCG@5 | 0.7827 |

### E2E（3 条 smoke）

| 指标 | 值 |
|---|---|
| completion rate | 100% |
| avg tool-chain coverage | 93.33% |
| specialist coverage | 100% |
| specialist correction | 5 |
| specialist error | 0 |

### RAGAS（3 条 full smoke，gemini-3.1-flash-lite）

| 指标 | 值 |
|---|---|
| faithfulness | 0.6170 |
| context_recall | 0.6667 |
| answer_correctness | 0.1135 |

> precision@5=0.18 是因为每题通常只有 1 条黄金法条（理论上限 0.2），非召回差；hit@1=0.6 但 hit@5=0.9 说明召回够、首位排序待优化。RAGAS 的 `answer_correctness` 偏低主要因 ground truth 过短、长答案被短 reference 低估——这是评测输入问题，非系统质量问题（见分层评测说明）。

---

## 致谢

- [LangChain](https://github.com/langchain-ai/langchain) / [LangGraph](https://github.com/langchain-ai/langgraph) — Workflow 框架
- [Qwen](https://github.com/QwenLM/Qwen3.5) — 基座模型
- [BAAI/bge](https://github.com/FlagOpen/FlagEmbedding) — Embedding 和 Reranker
- [FAISS](https://github.com/facebookresearch/faiss) — 向量数据库
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) — GGUF 推理引擎
- [RAGAS](https://github.com/explodinggradients/ragas) — RAG 评测
