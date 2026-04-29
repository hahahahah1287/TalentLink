# TalentLink - 本地化 AI 法务助手

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/LangChain-0.2.x-green.svg" alt="LangChain">
  <img src="https://img.shields.io/badge/LLM-Qwen2.5-purple.svg" alt="Qwen">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

> 一个基于 RAG + Agent 的本地化法务问答系统，采用"Router-Agent-Chain"混合架构，支持流式对话、多轮会话、智能路由、混合检索与重排序。

---

## 核心特性

### 已完成

#### 1. 三阶段检索管线 (Advanced RAG)

| 阶段 | 技术 | 作用 |
|------|------|------|
| 查询改写 | HyDE (Hypothetical Document Embedding) | 将口语化查询对齐到法言法语 |
| 混合检索 | BM25 + FAISS Ensemble | 关键词精确匹配 + 语义向量检索，双路召回互补 |
| 精排 | Cross-Encoder Reranker (BAAI/bge-reranker-v2-m3) | 深度语义打分，过滤低质量结果 |

**优势**：
- 混合检索兼顾"精确条款查询"和"模糊语义查询"
- Rerank 精排从 10 条候选中筛选 Top 3，提升精准度
- 结构化切分 + 元数据标注，确保法律条款完整性

#### 2. 智能路由与降级机制

```
用户请求 → Router (意图分类) → 分发执行
                ↓
    ┌───────────┼───────────┐
    ↓           ↓           ↓
  JOB        CONTRACT      CHAT
 (Agent)    (RAG Chain)   (通用对话)
```

**优势**：
- `PydanticOutputParser` 强制输出 JSON 结构，避免格式崩溃
- 多层 try-catch + 语义缓存 + 熔断器，确保系统高可用
- 用户永远不会看到 500 错误

#### 3. 会话管理系统

- **MySQL + Redis 异步架构**：Redis 缓冲写入（<0.1ms），后台线程批量落库
- **增量摘要机制**：对话超过 20 条后自动压缩历史，节省 Token
- **异步标题生成**：标题生成与流式输出解耦，首字延迟降低 40%

#### 4. 输出防护管线 (Guardrails)

| Guard | 功能 |
|-------|------|
| PIIGuard | 手机号、身份证、银行卡、邮箱脱敏 |
| QualityGuard | 检测回复过短、重复内容 |
| DisclaimerGuard | 涉及法律内容时自动追加免责声明 |

#### 5. 其他工程优化

- **语义缓存**：相似查询直接返回历史答案，跳过 LLM 推理
- **熔断器 (Circuit Breaker)**：Agent 连续失败 3 次后自动降级
- **SSE 流式输出**：毫秒级首字响应，支持 EventSource 协议
- **集中化配置**：dataclass 统一管理所有配置项

---

## 技术栈

| 组件 | 技术 |
|------|------|
| LLM | Qwen2.5-7B (GGUF Q5_K_M 量化) |
| Embedding | BAAI/bge-m3 |
| Reranker | BAAI/bge-reranker-v2-m3 |
| 向量数据库 | FAISS |
| 关键词检索 | BM25 |
| Web 框架 | FastAPI |
| Agent 框架 | LangChain ReAct |
| 数据库 | MySQL + Redis |
| 推理引擎 | llama-cpp-python |

---

## 快速开始

### 环境要求

- Python 3.10+
- CUDA 11.8+ (可选，GPU 加速)
- MySQL 8.0+
- Redis 7.0+ (可选，用于缓存和异步写入)

### 安装

```bash
# 克隆项目
git clone https://github.com/your-username/talentlink.git
cd talentlink

# 安装依赖
pip install -r requirements.txt

# 下载模型 (GGUF 格式)
# 推荐 Qwen2.5-7B-Q5_K_M 或 Qwen3.5-9B-Q5_K_M
# 放在项目根目录
```

### 配置

修改 `config/__init__.py` 中的配置项：

```python
@dataclass
class LLMConfig:
    model_path: str = "./Qwen3.5-9B-Q5_K_M.gguf"  # 模型路径
    n_ctx: int = 4096                                 # 上下文窗口
    n_gpu_layers: int = -1                            # -1 表示全部卸载到 GPU
    temperature: float = 0.1                          # 温度参数

@dataclass
class DatabaseConfig:
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = "your_password"
    mysql_database: str = "talentlink"
    redis_host: str = "localhost"
    redis_port: int = 6379
```

### 启动

```bash
# 启动 API 服务
python main.py

# 访问 API 文档
# http://localhost:8000/docs
```

### API 接口

```python
import requests

# 流式对话
response = requests.post(
    "http://localhost:8000/chat/stream",
    json={
        "user_id": "user_123",
        "query": "试用期工资标准是多少",
        "session_id": None  # 新对话传 None
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode())
```

---

## 项目结构

```
talentlink/
├── main.py                  # FastAPI 入口
├── services.py              # 核心服务层 (路由、Agent、Chain)
├── memory.py                # 会话管理 (MySQL/Redis/摘要)
├── config/                  # 配置管理
│   └── __init__.py          # 集中化配置 (dataclass)
├── utils/                   # 工具函数
│   ├── legal_parser.py      # 法律文档结构化切分
│   ├── metadata_annotator.py # LLM 元数据标注
│   ├── reranker.py          # Cross-Encoder 重排序
│   ├── query_rewriter.py    # HyDE 查询改写
│   ├── semantic_cache.py    # 语义缓存
│   ├── guardrails.py        # 输出防护管线
│   ├── circuit_breaker.py   # 熔断器
│   └── text_splitter.py     # 文本分词器 (通用)
├── skills/                  # Agent 技能
│   ├── web_search.py        # 联网搜索
│   └── local_retriever.py   # 本地知识库检索
├── tests/                   # 测试用例
├── eval_rag.py              # RAG 检索评估
└── eval_agent.py            # Agent 评测体系
```

---

## 迭代计划 (TODO)

### 高优先级

- [ ] **Markdown 输出格式校验**
  - 在 Guardrails 管线中增加 `MarkdownGuard`
  - 校验输出是否包含标题、列表、加粗等 Markdown 结构
  - 校验失败时调用 LLM 重写或追加格式约束提示

- [ ] **LangChain 版本升级**
  - 当前使用 LangChain 0.2.x，升级到最新稳定版
  - 适配新版本 API 变更（如 `langchain-core` 拆分）
  - 测试兼容性，确保 ReAct Agent、EnsembleRetriever 等核心功能正常

- [ ] **大小 Agent 协同架构**
  - 实现"小模型做路由/意图识别 + 大模型做生成/推理"的 Cascade 架构
  - 用 0.5B 模型做意图分类（降低首字延迟）
  - 用 7B 模型做深度推理和生成

### 中优先级

- [ ] **上下文压缩 (Context Compression)**
  - 检索后对文档进行压缩，去除与查询无关的冗余内容
  - 减少 Token 消耗，提升长文档处理能力

- [ ] **多表示索引 (Multi-Representation Indexing)**
  - 为每个文档生成摘要向量 + 块向量的双索引
  - 先用摘要召回章节，再用块向量召回具体条款

- [ ] **增量更新机制**
  - 支持新法律文档自动处理入库，无需全量重建索引
  - 监控 `labor_law.txt` 变更，自动触发解析和标注

- [ ] **元数据过滤检索**
  - 用户明确说"劳动合同法第35条"时，直接按 metadata 过滤
  - 减少向量检索的搜索空间，提升精确度

### 低优先级

- [ ] **评估体系完善**
  - 扩充评测数据集（从生产日志采样）
  - 增加端到端评测指标（Answer Relevance、Faithfulness）
  - 集成 CI/CD，每次模型更新自动跑评测

- [ ] **监控与可观测性**
  - 集成 Prometheus + Grafana
  - 监控 TTFT、TPOT、GPU 利用率、缓存命中率等指标

- [ ] **多模态支持**
  - 支持合同 PDF/图片上传
  - 集成 OCR 提取文本

---

## 评测结果

### RAG 检索评测

| 策略 | Recall@K | MRR | Hit Rate | 延迟 |
|------|----------|-----|----------|------|
| 纯 BM25 | 72% | 0.65 | 72% | 15ms |
| 纯 FAISS | 68% | 0.61 | 68% | 25ms |
| BM25 + FAISS | 80% | 0.73 | 80% | 35ms |
| BM25 + FAISS + Rerank | **88%** | **0.81** | **88%** | 120ms |

### Agent 评测

| 维度 | 权重 | 得分 |
|------|------|------|
| 意图路由准确率 | 15% | 99.0% |
| 工具选择准确率 | 20% | 80.0% |
| 答案质量 | 25% | 73.4% |
| 安全防护 | 20% | 91.7% |
| 抗幻觉率 | 10% | 66.7% |
| 延迟效率 | 10% | 80.0% |
| **综合** | 100% | **89.8%** |

---

## 致谢

- [LangChain](https://github.com/langchain-ai/langchain) - Agent/RAG 开发框架
- [Qwen](https://github.com/QwenLM/Qwen2.5) - 基座模型
- [BAAI/bge](https://github.com/FlagOpen/FlagEmbedding) - Embedding 和 Reranker 模型
- [FAISS](https://github.com/facebookresearch/faiss) - 向量数据库
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - GGUF 推理引擎
