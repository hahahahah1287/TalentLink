# -*- coding: utf-8 -*-
"""
Real RAGAS evaluation with Gemini / Google AI Studio as evaluator LLM.

This script uses the actual `ragas.evaluate(...)` API and RAGAS metrics.  The
project's article-level deterministic metrics still live in
`tests/eval_legal_retrieval.py` as legal-domain supplements, but this file is the
real RAGAS chain and can be described as RAGAS evaluation in resume/interviews.

Usage:
  pip install -r requirements-eval.txt
  GOOGLE_API_KEY=... python tests/eval_ragas.py \
    --e2e-report tests/reports/e2e.json \
    --output tests/reports/ragas_gemini.json

Input report format: JSON with `details`; each row should contain question,
final_answer/answer, contexts/law_context/evidence_items and ground_truth.
"""
import argparse
import asyncio
import json
import os
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.skill_result import render_skill_result_for_prompt


def _load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE pairs without adding a runtime dependency."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


_load_dotenv(Path(__file__).parent.parent / ".env")


def _install_ragas_vertexai_import_shim() -> None:
    """Keep RAGAS importable with newer langchain-community releases.

    RAGAS 0.3.x still imports `langchain_community.chat_models.vertexai` at
    module import time, while recent langchain-community no longer ships that
    module.  TalentLink evaluates with Gemini through google-genai, not VertexAI,
    so a minimal class is enough to satisfy RAGAS' isinstance checks.
    """
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return
    try:
        from langchain_core.language_models import BaseLanguageModel
    except Exception:
        return

    shim = types.ModuleType(module_name)

    class ChatVertexAI(BaseLanguageModel):
        pass

    shim.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = shim


def _load_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data.get("legal"), dict):
        return data["legal"].get("details", [])
    return data.get("details", [])


def _contexts_from_row(row: Dict[str, Any]) -> List[str]:
    contexts = row.get("contexts")
    if isinstance(contexts, list) and contexts:
        return [str(c) for c in contexts if c]

    rendered: List[str] = []

    evidence_items = row.get("evidence_items") or []
    for item in evidence_items:
        source = item.get("source") or {}
        citation = source.get("canonical_citation") or item.get("article_id") or "未知来源"
        text = (item.get("text") or "").strip()
        rendered.append(f"【法条证据】{citation}\n{text}" if text else f"【法条证据】{citation}")

    law_context = row.get("law_context")
    if law_context:
        rendered.append(f"【检索法条上下文】\n{law_context}")

    contract_text = row.get("contract_text")
    if contract_text:
        rendered.append(f"【合同原文】\n{contract_text}")

    skill_outputs = row.get("skill_outputs") or {}
    if isinstance(skill_outputs, dict):
        for skill_name, raw_output in skill_outputs.items():
            rendered_skill = render_skill_result_for_prompt(skill_name, raw_output)
            if rendered_skill:
                rendered.append(f"【专家分析：{skill_name}】\n{rendered_skill}")

    return [part for part in rendered if part]


def _normalize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, 1):
        question = row.get("question") or row.get("query") or row.get("input") or ""
        answer = row.get("answer") or row.get("final_answer_full") or row.get("final_answer") or row.get("output") or ""
        ground_truth = row.get("ground_truth") or row.get("reference") or row.get("expected_answer") or ""
        contexts = _contexts_from_row(row)
        if not question or not answer or not contexts:
            continue
        normalized.append({
            "id": row.get("id", f"sample_{idx:03d}"),
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        })
    return normalized


def _build_llm(model: str, min_interval_s: float = 13.0):
    _install_ragas_vertexai_import_shim()

    from google import genai
    from google.genai import types as genai_types
    from langchain_core.outputs import Generation, LLMResult
    from ragas.llms.base import BaseRagasLLM

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_KEY")
    if not api_key:
        raise RuntimeError("请设置 GOOGLE_API_KEY、GEMINI_API_KEY 或 GEMINI_KEY")
    client = genai.Client(api_key=api_key)
    rate_lock = threading.Lock()
    last_call_at = 0.0

    class GeminiRagasLLM(BaseRagasLLM):
        def _call(self, prompt_text: str, temperature: Optional[float] = None) -> str:
            nonlocal last_call_at
            with rate_lock:
                elapsed = time.monotonic() - last_call_at
                if elapsed < min_interval_s:
                    time.sleep(min_interval_s - elapsed)
                last_call_at = time.monotonic()
            response = client.models.generate_content(
                model=model,
                contents=prompt_text,
                config=genai_types.GenerateContentConfig(
                    temperature=0.0 if temperature is None else temperature,
                    candidate_count=1,
                ),
            )
            return response.text or ""

        def generate_text(self, prompt, n: int = 1, temperature: float = 0.01, stop=None, callbacks=None):
            generations = []
            for _ in range(n):
                generations.append(Generation(text=self._call(prompt.to_string(), temperature)))
            return LLMResult(generations=[generations])

        async def agenerate_text(self, prompt, n: int = 1, temperature: Optional[float] = 0.01, stop=None, callbacks=None):
            generations = []
            for _ in range(n):
                text = await asyncio.to_thread(self._call, prompt.to_string(), temperature)
                generations.append(Generation(text=text))
            return LLMResult(generations=[generations])

        def is_finished(self, response: LLMResult) -> bool:
            return True

    return GeminiRagasLLM()


def _build_metrics(llm, selected: Optional[List[str]] = None):
    _install_ragas_vertexai_import_shim()
    from ragas.metrics import AnswerCorrectness, ContextPrecision, ContextRecall, Faithfulness

    all_metrics = {
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
        "faithfulness": Faithfulness(llm=llm),
        # Use factual correctness only: this keeps the metric fully LLM-judged and
        # avoids pulling a second evaluator embedding provider into the eval stack.
        "answer_correctness": AnswerCorrectness(llm=llm, weights=[1.0, 0.0]),
    }
    if not selected:
        return list(all_metrics.values())
    unknown = [name for name in selected if name not in all_metrics]
    if unknown:
        raise ValueError(f"未知 RAGAS metric: {unknown}. 可选: {sorted(all_metrics)}")
    return [all_metrics[name] for name in selected]


def main():
    parser = argparse.ArgumentParser(description="Real RAGAS evaluation using Gemini evaluator LLM")
    parser.add_argument("--e2e-report", required=True, help="E2E JSON report with details rows")
    parser.add_argument("--output", default="tests/reports/ragas_gemini.json")
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
    parser.add_argument("--offset", type=int, default=0, help="跳过前 N 条样本，便于限额下分批续跑")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--metrics",
        default=os.getenv("RAGAS_METRICS", ""),
        help="逗号分隔：context_precision,context_recall,faithfulness,answer_correctness；空值=全部",
    )
    parser.add_argument("--max-workers", type=int, default=int(os.getenv("RAGAS_MAX_WORKERS", "1")))
    parser.add_argument("--min-call-interval", type=float, default=float(os.getenv("GEMINI_MIN_CALL_INTERVAL", "13")))
    parser.add_argument("--max-retries", type=int, default=int(os.getenv("RAGAS_MAX_RETRIES", "3")))
    parser.add_argument("--max-wait", type=int, default=int(os.getenv("RAGAS_MAX_WAIT", "30")))
    args = parser.parse_args()

    rows = _normalize_rows(_load_rows(args.e2e_report))
    if args.offset > 0:
        rows = rows[args.offset:]
    if args.limit > 0:
        rows = rows[:args.limit]
    if not rows:
        raise RuntimeError("没有可评估样本：需要 question/answer/contexts 至少三个字段")

    _install_ragas_vertexai_import_shim()
    from datasets import Dataset
    from ragas import evaluate
    from ragas.run_config import RunConfig

    from config import AppConfig
    from utils.embeddings import TransformerEmbeddings

    dataset = Dataset.from_list([
        {
            "user_input": row["question"],
            "response": row["answer"],
            "retrieved_contexts": row["contexts"],
            "reference": row["ground_truth"],
        }
        for row in rows
    ])

    evaluator_llm = _build_llm(args.model, min_interval_s=args.min_call_interval)
    selected_metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    metrics = _build_metrics(evaluator_llm, selected_metrics)
    config = AppConfig()
    evaluator_embeddings = TransformerEmbeddings(
        model_name=config.embedding.model_name,
        device=config.embedding.device,
        normalize_embeddings=config.embedding.normalize_embeddings,
        local_files_only=True,
    )
    run_config = RunConfig(
        timeout=180,
        max_retries=args.max_retries,
        max_wait=args.max_wait,
        max_workers=args.max_workers,
    )
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        embeddings=evaluator_embeddings,
        run_config=run_config,
        batch_size=1,
        raise_exceptions=True,
    )

    # RAGAS versions expose results slightly differently; keep this tolerant.
    if hasattr(result, "to_pandas"):
        details = result.to_pandas().to_dict(orient="records")
    elif hasattr(result, "scores"):
        details = result.scores
    else:
        details = json.loads(str(result)) if str(result).startswith("{") else []

    summary: Dict[str, Any] = {"model": args.model, "count": len(rows)}
    try:
        summary.update({k: float(v) for k, v in dict(result).items()})
    except Exception:
        pass
    metric_keys = ["context_precision", "context_recall", "faithfulness", "answer_correctness"]
    for key in metric_keys:
        values = []
        for row in details:
            value = row.get(key) if isinstance(row, dict) else None
            if isinstance(value, (int, float)) and value == value:
                values.append(float(value))
        if values and key not in summary:
            summary[key] = sum(values) / len(values)

    report = {
        "summary": summary,
        "sample_ids": [row["id"] for row in rows],
        "details": details,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"📄 RAGAS report saved: {output_path}")


if __name__ == "__main__":
    main()
