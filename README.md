# TalentLink - 本地化 AI 法务助手

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/LangGraph-0.2.x-green.svg" alt="LangGraph">
  <img src="https://img.shields.io/badge/LLM-Qwen3.5-purple.svg" alt="Qwen">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

> 基于 LangGraph 状态图的本地化法务 AI 系统。两条 workflow 覆盖合同审查和通用研究，Review Agent 自主审查输出质量，全程本地推理，数据不出域。

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

---

## 快速开始

### 环境要求

- Python 3.13+
- CUDA 11.8+ (可选，GPU 加速)
- MySQL 8.0+
- Redis 7.0+ (可选，用于缓存和异步写入)

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
├── config/
│   └── __init__.py              # 集中化配置 (dataclass)
├── workflows/
│   ├── __init__.py
│   ├── contract_review.py       # 合同审查 workflow
│   ├── research_agent.py        # 研究型任务 workflow
│   └── review_agent.py          # Review Agent（输出审查）
├── utils/
│   ├── retrieval_service.py     # 统一检索服务（HyDE + 混合检索 + Rerank）
│   ├── reranker.py              # Cross-Encoder 重排序
│   ├── query_rewriter.py        # HyDE 查询改写
│   ├── semantic_cache.py        # 语义缓存
│   ├── guardrails.py            # Guard 工具实现
│   ├── legal_parser.py          # 法律文档结构化切分
│   ├── metadata_annotator.py    # LLM 元数据标注
│   ├── state.py                 # LangGraph AppState 定义
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
- [Qwen](https://github.com/QwenLM/Qwen3.5) - 基座模型
- [BAAI/bge](https://github.com/FlagOpen/FlagEmbedding) - Embedding 和 Reranker
- [FAISS](https://github.com/facebookresearch/faiss) - 向量数据库
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - GGUF 推理引擎
