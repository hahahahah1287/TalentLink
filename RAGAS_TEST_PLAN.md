# TalentLink RAGAS 测试链路方案（真实 RAGAS + 法务确定性补充指标）

> 结论：既然现在已经接入 Gemini / Google AI Studio 作为第三方 evaluator LLM，就不再叫 “RAGAS-style”。
>
> 新方案是：**真实 RAGAS Evaluation** 作为主评测链路，法务场景特有的 article-level 指标作为补充评测链路。

---

## 1. 总体定位

项目评测拆成两条互补链路：

```text
A. tests/eval_ragas.py
   真实调用 ragas.evaluate(...)
   使用 Gemini evaluator LLM
   评估 faithfulness / answer correctness / context precision / context recall

B. tests/eval_legal_retrieval.py
   法务确定性补充指标
   评估 expected_articles / Hit@K / MRR / nDCG / Citation Support / Tool Coverage
```

为什么要两条链路：

1. **RAGAS** 能评估语义质量：回答是否忠实、是否相关、是否与标准答案一致。
2. **法务确定性指标** 能评估硬约束：有没有检索到正确法条、有没有引用本轮证据外的条文、工具链有没有按预期执行。
3. 法律场景不能只看 LLM Judge 分数，因为 Judge 也可能被流畅但错误的回答骗过。
4. 所以最终说法应是：

> 本项目接入真实 RAGAS 评测链路，并补充 article-level legal metrics，用于定位问题发生在召回、引用、工具链还是生成质量。

---

## 2. 评测分层

```text
Layer 0: Dataset Quality
Layer 1: RAGAS Evaluation
Layer 2: Article-level Retrieval Metrics
Layer 3: Citation Grounding Metrics
Layer 4: Route / Skill / Guard Metrics
Layer 5: E2E Regression Report
```

| 层级 | 目标 | 主要脚本 | 是否需要 LLM Judge |
|---|---|---|---|
| Layer 0 | 数据集字段、expected_articles、法源一致性检查 | `eval_dataset_quality.py`（后续） | 否 |
| Layer 1 | 真实 RAGAS 指标 | `tests/eval_ragas.py` | 是，Gemini |
| Layer 2 | 检索是否命中正确法条 | `tests/eval_legal_retrieval.py` | 否 |
| Layer 3 | 引用是否被 evidence 支持 | `eval_citation_grounding.py`（后续） | 否 |
| Layer 4 | route / skill / guard 是否按预期执行 | `tests/eval_e2e.py` / `tests/test_skills.py` | 否 |
| Layer 5 | 汇总报告与回归趋势 | reports JSON / 后续 TEST_REPORT | 视指标而定 |

---

## 3. 真实 RAGAS 主链路

### 3.1 脚本

当前主脚本：

```text
tests/eval_ragas.py
```

它做的事情：

1. 读取 E2E report。
2. 抽取 RAGAS 所需字段：
   - `question`
   - `answer`
   - `contexts`
   - `ground_truth`
3. 使用 Gemini / Google AI Studio 作为 evaluator LLM。
4. 调用真实的：

```python
ragas.evaluate(dataset=dataset, metrics=metrics)
```

### 3.2 RAGAS 指标

第一版使用：

```python
ContextPrecision
ContextRecall
Faithfulness
AnswerCorrectness
```

对应含义：

| RAGAS 指标 | 解释 |
|---|---|
| `ContextPrecision` | 检索上下文中有多少内容对回答有用 |
| `ContextRecall` | 标准答案所需信息是否被上下文覆盖 |
| `Faithfulness` | 回答中的断言是否能被上下文支持 |
| `AnswerCorrectness` | 回答与标准答案是否一致 |

### 3.3 命令

安装评测依赖：

```bash
pip install -r requirements-eval.txt
```

运行真实 RAGAS：

```bash
GOOGLE_API_KEY=你的key python tests/eval_ragas.py \
  --e2e-report tests/reports/e2e.json \
  --output tests/reports/ragas_gemini.json
```

也兼容：

```bash
GEMINI_API_KEY=你的key python tests/eval_ragas.py \
  --e2e-report tests/reports/e2e.json \
  --output tests/reports/ragas_gemini.json
```

小样本 smoke：

```bash
GOOGLE_API_KEY=你的key python tests/eval_ragas.py \
  --e2e-report tests/reports/e2e.json \
  --limit 5 \
  --output tests/reports/ragas_smoke.json
```

---

## 4. E2E report 字段要求

RAGAS 需要的最小字段：

```json
{
  "question": "劳动合同中约定试用期一年合法吗？",
  "final_answer": "不合法。根据...",
  "law_context": "【E1】《中华人民共和国劳动法》第二十一条...",
  "ground_truth": "不合法。根据《劳动法》第二十一条，试用期最长不得超过六个月。"
}
```

`tests/eval_ragas.py` 会自动兼容这些字段名：

| RAGAS 字段 | 可读取字段 |
|---|---|
| question | `question` / `query` / `input` |
| answer | `answer` / `final_answer` / `output` |
| contexts | `contexts` / `evidence_items` / `law_context` |
| ground_truth | `ground_truth` / `reference` / `expected_answer` |

如果有 `evidence_items`，脚本会优先把它渲染成 RAGAS contexts。

---

## 5. 法务确定性补充指标

真实 RAGAS 解决“语义质量”问题，但法律项目还需要硬指标。

补充脚本：

```text
tests/eval_legal_retrieval.py
```

它评估：

- Hit@K
- Recall@K
- Precision@K
- MRR
- nDCG@K
- context token precision / recall（辅助）
- missing refs
- unexpected refs

运行：

```bash
python tests/eval_legal_retrieval.py \
  --limit 100 \
  --top-k 5 \
  --no-hyde \
  --output tests/reports/legal_retrieval.json
```

注意：这个脚本依赖本地 embedding / FAISS，因此需要先安装主项目依赖：

```bash
pip install -r requirements.txt
```

---

## 6. 为什么真实 RAGAS 后仍保留 deterministic metrics

面试时可以这样解释：

> RAGAS 能评估回答质量，但法律系统还需要法条级硬约束。比如 Gemini 可能认为一个答案语义上合理，但它引用了本轮 evidence 里没有的《劳动合同法》第 X 条，这在法务系统里是硬错误。所以我把 RAGAS 作为语义评测主链路，同时保留 article-level deterministic metrics，用来检查检索命中、引用支撑和工具链覆盖。

这个说法比单纯“我用了 RAGAS”更强，因为它体现了你知道：

- 通用 RAG 指标的价值；
- 法务垂直场景的额外约束；
- LLM Judge 的边界；
- 如何把评测结果定位到 retrieval / generation / guard / skill。

---

## 7. 推荐报告结构

```text
tests/reports/
  ragas_gemini.json             # 真实 RAGAS 指标
  legal_retrieval.json          # 法条级检索指标
  citation_grounding.json       # 引用支撑指标（后续）
  e2e.json                      # 端到端回归报告
```

最终报告建议汇总成：

```json
{
  "ragas": {
    "faithfulness": 0.91,
    "answer_correctness": 0.86,
    "context_precision": 0.82,
    "context_recall": 0.88
  },
  "legal_retrieval": {
    "hit@5": 0.93,
    "mrr": 0.79,
    "ndcg@5": 0.82
  },
  "citation_grounding": {
    "citation_support_rate": 0.96,
    "unsupported_citation_avg": 0.04
  },
  "tooling": {
    "tool_required_coverage": 0.95,
    "guard_trigger_accuracy": 0.90
  }
}
```

上面的数字只是报告格式示例，不是当前真实结果。

---

## 8. 简历表达

推荐写法：

> 接入真实 RAGAS 评测链路，使用 Gemini / Google AI Studio 作为 evaluator LLM，评估法务 RAG 的 Faithfulness、Answer Correctness、Context Precision 和 Context Recall；同时设计 article-level deterministic legal metrics，基于 expected_articles、evidence trace 和 citation matching 量化 Hit@K、MRR、nDCG、Citation Support 与 Tool Coverage。

更短一点：

> 构建 Gemini-backed RAGAS 评测链路，并补充法条级 deterministic metrics，实现法务 RAG 的语义质量、检索命中、引用支撑和工具链覆盖评估。

---

## 9. 和当前代码的对应关系

| 能力 | 文件 |
|---|---|
| 真实 RAGAS + Gemini evaluator | `tests/eval_ragas.py` |
| 法条级 deterministic retrieval eval | `tests/eval_legal_retrieval.py` |
| RAGAS 评测依赖 | `requirements-eval.txt` |
| article-level evidence | `utils/evidence.py` |
| law/article pair CitationGuard | `utils/guardrails.py` |
| E2E 输出来源 | `tests/eval_e2e.py` |
| SkillResult / provenance | `utils/skill_result.py` |

---

## 10. 当前注意事项

1. `tests/eval_ragas.py` 是真实 RAGAS 链路，需要：

```bash
pip install -r requirements-eval.txt
```

2. `tests/eval_legal_retrieval.py` 是本地检索链路，需要：

```bash
pip install -r requirements.txt
```

3. 当前项目不再建议说 “RAGAS-style”。应该说：

```text
真实 RAGAS evaluation + legal deterministic metrics
```
