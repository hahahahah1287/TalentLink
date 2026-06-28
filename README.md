# TalentLink - 本地化 AI 法务助手

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/LangGraph-0.2.x-green.svg" alt="LangGraph">
  <img src="https://img.shields.io/badge/LLM-Qwen3.5-purple.svg" alt="Qwen">
  <img src="https://img.shields.io/badge/Langfuse-Observability-orange.svg" alt="Langfuse">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

> 基于 LangGraph 状态图的本地化法务 AI 系统。两条 workflow 覆盖合同审查和通用研究，Review Agent 自主审查输出质量，全程本地推理，数据不出域。集成 Langfuse 实现全链路可观测性。

---

## 架构

```
用户请求 → WorkflowService（安全检测 → 语义缓存 → 会话管理）
                ↓
        scene 参数决定进入哪条 workflow
        ┌───────────┴───────────┐
        ↓                       ↓
   Contract Review          Research Agent
  (固定流程，快速)        (Planner 动态规划)

  retrieve                plan（选工具）
     ↓                    ↓
  analyze              execute（循环）
     ↓                    ↓
   review ←────────── synthesize
     ↓                    ↓
  (approve)            review
     ↓                    ↓
   END               (approve → END)
                       (revise → re_synthesize → review)
```

### Contract Review（合同审查）

固定流程 `retrieve → analyze → review`，不需要 Planner 规划步骤：

1. **retrieve** — HyDE 查询改写 + BM25/FAISS 混合检索 + Cross-Encoder Rerank
2. **analyze** — LLM 基于检索到的法律条文分析合同条款
3. **review** — Review Agent 自主审查输出，发现引用问题触发 re_synthesize 重试

### Research Agent（研究型任务）

动态流程 `plan → execute → synthesize → review`，Planner 根据问题自动选择工具：

1. **plan** — LLM 规划执行步骤（legal_search / web_search / job_search）
2. **execute** — 循环执行计划中的每一步
3. **synthesize** — 综合所有检索结果生成最终答案
4. **review** — Review Agent 审查，发现问题触发 re_synthesize 重试

### Review Agent（输出审查）

独立的 ReAct Agent，把 Guardrails 包装为工具，自主决定审查策略：

| 工具 | 功能 |
|------|------|
| `pii_check` | 检测并脱敏个人信息（手机号、身份证、银行卡、邮箱） |
| `citation_check` | 验证引用的法律条文是否存在于检索结果中 |
| `quality_check` | 检查回复是否过短或有重复内容 |
| `add_disclaimer` | 涉及法律内容时自动追加免责声明 |

与旧的固定 Guardrails 管线的区别：Review Agent 自主决定调用哪些工具、按什么顺序，而不是按固定顺序跑所有检查。

---

## 核心特性

### 三阶段检索管线

| 阶段 | 技术 | 作用 |
|------|------|------|
| 查询改写 | HyDE (Hypothetical Document Embedding) | 将口语化查询对齐到法言法语 |
| 混合检索 | BM25 + FAISS Ensemble (k=8) | 关键词精确匹配 + 语义向量检索 |
| 精排 | Cross-Encoder Reranker (top_k=5, threshold=0.3) | 深度语义打分，过滤低质量结果 |

### 法律文档结构化处理

- **树状切分**：按 篇→章→节→条 结构切分，保证条款完整性
- **元数据标注**：LLM 自动标注每条的法律领域、关键词、适用场景
- **带元数据索引**：FAISS 索引携带元数据，支持过滤检索

### Langfuse 可观测性

集成 [Langfuse](https://langfuse.com/) 实现 LLM 调用的全链路追踪，无需侵入业务代码。

- **自动 Trace**：通过 `LangfuseLLMCallback` 回调，自动捕获每次 LLM 调用的输入、输出、Token 用量和延迟
- **Workflow 级 Span**：每个 workflow 节点（retrieve / analyze / review 等）自动创建 Span，支持端到端链路分析
- **无侵入集成**：回调通过 LangChain 的 `callbacks` 机制注入，workflow 代码无需改动

配置方式：

```bash
# 环境变量（推荐）
export LANGFUSE_PUBLIC_KEY="pk-..."
export LANGFUSE_SECRET_KEY="sk-..."
export LANGFUSE_HOST="http://localhost:3000"   # 自部署或 Langfuse Cloud
```

```python
# config/__init__.py 中的相关配置
@dataclass
class LangfuseConfig:
    enabled: bool = True
    host: str = "http://localhost:3000"
    public_key: str = ""       # 或通过环境变量 LANGFUSE_PUBLIC_KEY
    secret_key: str = ""       # 或通过环境变量 LANGFUSE_SECRET_KEY
    flush_at: int = 15         # 批量上报阈值
    flush_interval: float = 1.0  # 定时上报间隔（秒）
```

Langfuse 提供以下端点用于查看追踪数据：

| 端点 | 用途 |
|------|------|
| `GET /api/public/traces` | 查询所有 Trace 记录 |
| `GET /api/public/observations` | 查询 Span/Span 详情 |
| `GET /api/public/scores` | 查询评估分数 |
| Web UI `/project/traces` | 可视化链路追踪看板 |

详细部署指南见 `LANGFUSE_SETUP.md`。

### Redis 结果缓存（WorkflowCheckpoint）

基于 Redis 的 workflow 结果缓存，相同查询直接返回历史结果，跳过整个 workflow 执行。

- **缓存键**：`checkpoint:{workflow}:{query_hash}`，按 workflow 类型 + 查询内容的 MD5 哈希索引
- **TTL 过期**：默认 1 小时，支持手动失效和批量清除
- **智能写入**：只缓存有效结果（`final_answer` 非空且长度 > 10），避免缓存异常输出
- **命中率统计**：内置 hits/misses 计数器，支持运行时监控
- **无缓存降级**：Redis 不可用时自动降级为无缓存模式，不影响正常服务

```python
# WorkflowService 中的使用方式
from utils import WorkflowCheckpoint

checkpoint = WorkflowCheckpoint(redis_client, ttl=3600)

# 执行前检查缓存
cached = checkpoint.get("contract", query)
if cached:
    return cached

# 执行 workflow ...
result = await graph.ainvoke(state)

# 写入缓存
checkpoint.set("contract", query, result)
```

### ToolCallParserLLM — LLM Tool Call XML 解析修复

解决 ChatLlamaCpp + Qwen 3.5 等模型的 tool call 输出格式兼容问题。

**问题**：模型通过 `bind_tools` 学会了输出 tool call，但以纯文本 XML 格式输出在 `content` 字段，而非标准的 `tool_calls` 属性。导致 `create_react_agent` 的 Agent Loop 将 XML 当作最终回答返回，工具调用链路断裂。

**方案**：`ToolCallParserLLM` 包装层，在 `invoke`/`ainvoke` 后自动：
1. 检查 `AIMessage.tool_calls` 是否为空
2. 若为空，从 `content` 解析 XML 标签填充到 `tool_calls`
3. 清理 `content` 中的 XML 残留

```python
from utils import ToolCallParserLLM

raw_llm = ChatLlamaCpp(model_path=..., n_ctx=4096)
llm = ToolCallParserLLM(raw_llm)
agent = create_react_agent(llm, tools=[...])  # 自动受益，无需改动 Agent 代码
```

透明代理设计：`bind_tools`、`with_structured_output` 等方法透传到底层 LLM，`__getattr__` 兜底其余属性，对上层完全透明。

### 其他

- **语义缓存**：相似查询直接返回历史答案，跳过 LLM 推理
- **熔断器 (Circuit Breaker)**：连续失败自动降级
- **SSE 流式输出**：毫秒级首字响应
- **会话管理**：MySQL + Redis 异步架构，增量摘要压缩历史
- **安全检测**：Prompt Injection 检测拦截

---

## 技术栈

| 组件 | 技术 |
|------|------|
| LLM | Qwen3.5-9B (GGUF Q5_K_M 量化) |
| Embedding | BAAI/bge-m3 |
| Reranker | BAAI/bge-reranker-v2-m3 |
| 向量数据库 | FAISS |
| 关键词检索 | BM25 |
| 工作流编排 | LangGraph StateGraph |
| Agent 框架 | LangChain ReAct (create_react_agent) |
| Web 框架 | FastAPI |
| 数据库 | MySQL + Redis |
| 推理引擎 | llama-cpp-python |
| 可观测性 | Langfuse（LLM 调用追踪 + 链路分析） |

---

## 快速开始

### 环境要求

- Python 3.13+
- CUDA 11.8+ (可选，GPU 加速)
- MySQL 8.0+
- Redis 7.0+ (可选，用于缓存和异步写入)
- Langfuse (可选，用于可观测性，支持自部署或 Cloud)

### 安装

```bash
git clone https://github.com/your-username/talentlink.git
cd talentlink
pip install -r requirements.txt

# 下载 GGUF 模型放在项目根目录
# 推荐 Qwen3.5-9B-Q5_K_M
```

### 配置

修改 `config/__init__.py` 中的配置项：

```python
@dataclass
class LLMConfig:
    model_path: str = "./Qwen3.5-9B-Q5_K_M.gguf"
    n_ctx: int = 4096
    n_gpu_layers: int = -1        # -1 表示全部卸载到 GPU
    temperature: float = 0.1

@dataclass
class RetrievalConfig:
    bm25_weight: float = 0.4      # BM25 权重
    faiss_weight: float = 0.6     # FAISS 权重
    retrieval_k: int = 8          # 粗排候选数
    hyde_enabled: bool = True     # HyDE 查询改写
```

### 启动

```bash
python main.py
# 访问 http://localhost:8000/docs
```

### API 接口

```python
import requests

# 合同审查
response = requests.post(
    "http://localhost:8000/chat/stream",
    json={
        "user_id": "user_123",
        "query": "这份合同的试用期约定是否合法？",
        "scene": "contract",
        "contract_text": "试用期为6个月，劳动合同期限为1年。"
    },
    stream=True
)

# 通用研究
response = requests.post(
    "http://localhost:8000/chat/stream",
    json={
        "user_id": "user_123",
        "query": "劳动法第三十八条规定了什么？",
        "scene": "research"
    },
    stream=True
)
```

---

## 项目结构

```
talentlink/
├── main.py                      # FastAPI 入口
├── workflow_service.py          # LangGraph Workflow Service（核心服务层）
├── memory.py                    # 会话管理 (MySQL/Redis/摘要)
├── LANGFUSE_SETUP.md            # Langfuse 部署与配置指南
├── config/
│   └── __init__.py              # 集中化配置 (dataclass)
├── workflows/
│   ├── __init__.py
│   ├── contract_review.py       # 合同审查 workflow
│   ├── research_agent.py        # 研究型任务 workflow
│   ├── review_agent.py          # Review Agent（输出审查）
│   └── shared_nodes.py          # 共享 LangGraph 节点（synthesize 等）
├── skills/
│   ├── __init__.py
│   ├── base.py                  # Skill 基类
│   ├── registry.py              # 技能注册表（SkillRegistry）
│   ├── web_search.py            # 联网搜索 / 招聘搜索
│   ├── risk_clause_detector.py  # 风险条款识别
│   ├── compliance_check.py      # 合规性检查
│   ├── legal_term_explainer.py  # 法律术语解释
│   └── statute_checker.py       # 法条适用性检查
├── utils/
│   ├── __init__.py
│   ├── retrieval_service.py     # 统一检索服务（HyDE + 混合检索 + Rerank）
│   ├── reranker.py              # Cross-Encoder 重排序
│   ├── query_rewriter.py        # HyDE 查询改写
│   ├── semantic_cache.py        # 语义缓存
│   ├── guardrails.py            # Guard 工具实现
│   ├── legal_parser.py          # 法律文档结构化切分
│   ├── metadata_annotator.py    # LLM 元数据标注
│   ├── state.py                 # LangGraph AppState 定义
│   ├── checkpoint.py            # Redis 结果缓存（WorkflowCheckpoint）
│   ├── tool_call_parser.py      # ToolCallParserLLM（XML tool call 解析）
│   ├── langfuse_integration.py  # Langfuse 回调与 Trace 集成
│   └── tools/
│       ├── contract_tools.py    # 合同分析 LLM Chain
│       ├── guard_tools.py       # Guard → LangChain Tool 封装
│       └── job_tools.py         # 联网搜索 / 招聘搜索
├── services.py                  # 安全检测等辅助服务
├── tests/
│   ├── ragas_dataset.py         # RAG 评估数据集 (100 条)
│   ├── eval_ragas.py            # RAG 检索质量评估
│   ├── eval_e2e.py              # 全链路端到端评估
│   ├── eval_review_agent.py     # Review Agent 评估 (LangSmith)
│   └── TEST_REPORT.md           # 测试报告
└── labor_law.txt                # 法律知识库
```

---

## 评测结果

详见 `tests/TEST_REPORT.md`。

### RAG 检索质量（100 条）

| 指标 | Overall | Contract (n=43) | Other (n=57) |
|---|---|---|---|
| Context Precision | 0.6452 | 0.6454 | 0.6451 |
| Context Recall | 0.7976 | 0.7415 | 0.8400 |

### 全链路端到端

| 指标 | Contract (n=10) | Research (n=5) |
|---|---|---|
| Completion Rate | 1.00 | 1.00 |
| Answer Relevance | 0.77 | 0.78 |
| Tool Chain Coverage | 1.00 | 0.95 |
| Review Status Accuracy | 1.00 | — |
| Avg Latency | 34.2s | 30.3s |

---

## 致谢

- [LangChain](https://github.com/langchain-ai/langchain) / [LangGraph](https://github.com/langchain-ai/langgraph) - Agent/Workflow 框架
- [Langfuse](https://github.com/langfuse/langfuse) - LLM 可观测性平台
- [Qwen](https://github.com/QwenLM/Qwen3.5) - 基座模型
- [BAAI/bge](https://github.com/FlagOpen/FlagEmbedding) - Embedding 和 Reranker
- [FAISS](https://github.com/facebookresearch/faiss) - 向量数据库
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - GGUF 推理引擎
