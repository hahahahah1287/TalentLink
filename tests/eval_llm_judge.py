# -*- coding: utf-8 -*-
"""
Optional Gemini / Google AI Studio judge for E2E answer quality.

This script is intentionally separate from deterministic `eval_ragas.py`:
- CI and local-first architecture do not depend on an external judge.
- When GEMINI_API_KEY is present, Gemini can score faithfulness/relevancy/
  correctness against provided references, not against open-world memory.

Input format: an E2E JSON report with `details`, where each row preferably has:
  id/question/ground_truth/final_answer/law_context/evidence_items

Usage:
  GEMINI_API_KEY=... python tests/eval_llm_judge.py \
    --e2e-report tests/reports/e2e.json \
    --output tests/reports/llm_judge_gemini.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_MODEL = "gemini-2.0-flash"


JUDGE_PROMPT = """你是法务 RAG 评测裁判。请只基于【参考证据】和【标准答案】评价【模型回答】，不要使用开放世界知识补全。

请输出严格 JSON，不要 Markdown。字段：
{{
  "faithfulness": 0到1之间的小数,       // 回答中的事实是否被参考证据支持
  "answer_relevancy": 0到1之间的小数,   // 是否正面回答用户问题
  "answer_correctness": 0到1之间的小数, // 与标准答案关键结论是否一致
  "unsupported_claims": ["..."],        // 未被证据支持的重要断言
  "missing_points": ["..."],            // 相比标准答案遗漏的关键点
  "verdict": "pass" 或 "fail",
  "reason": "一句中文解释"
}}

【用户问题】
{question}

【参考证据】
{references}

【标准答案】
{ground_truth}

【模型回答】
{final_answer}
"""


def _load_report(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)
    if isinstance(report, list):
        return report
    return report.get("details", [])


def _references_from_row(row: Dict[str, Any]) -> str:
    evidence = row.get("evidence_items") or []
    if evidence:
        parts = []
        for item in evidence:
            source = item.get("source") or {}
            citation = source.get("canonical_citation") or item.get("article_id") or "未知来源"
            text = (item.get("text") or "").strip()
            parts.append(f"{citation}\n{text}")
        return "\n\n".join(parts)
    return row.get("law_context") or "（无参考证据）"


def _parse_json_response(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start:end + 1]
    return json.loads(text)


def _judge_with_genai(client, model: str, prompt: str) -> Dict[str, Any]:
    response = client.models.generate_content(model=model, contents=prompt)
    return _parse_json_response(getattr(response, "text", ""))


def main():
    parser = argparse.ArgumentParser(description="Optional Gemini LLM judge for E2E reports")
    parser.add_argument("--e2e-report", required=True)
    parser.add_argument("--output", default="tests/reports/llm_judge_gemini.json")
    parser.add_argument("--model", default=os.getenv("GEMINI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0, help="每条评测之间暂停，避免触发限流")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("请先设置 GEMINI_API_KEY 或 GOOGLE_API_KEY")

    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("缺少 google-genai 依赖：pip install google-genai") from exc

    rows = _load_report(args.e2e_report)
    if args.limit > 0:
        rows = rows[:args.limit]

    client = genai.Client(api_key=api_key)
    details: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows, 1):
        question = row.get("question") or row.get("input") or ""
        final_answer = row.get("final_answer") or row.get("answer") or row.get("output") or ""
        ground_truth = row.get("ground_truth") or row.get("expected_answer") or ""
        prompt = JUDGE_PROMPT.format(
            question=question,
            references=_references_from_row(row),
            ground_truth=ground_truth,
            final_answer=final_answer,
        )
        print(f"[{idx}/{len(rows)}] Gemini judging: {question[:50]}...")
        try:
            judge = _judge_with_genai(client, args.model, prompt)
        except Exception as exc:
            judge = {"error": str(exc), "verdict": "error"}
        details.append({
            "id": row.get("id", f"sample_{idx:03d}"),
            "question": question,
            "judge": judge,
        })
        if args.sleep:
            time.sleep(args.sleep)

    def avg_score(key: str) -> float:
        vals = [d["judge"].get(key) for d in details if isinstance(d.get("judge"), dict)]
        vals = [float(v) for v in vals if isinstance(v, (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    summary = {
        "model": args.model,
        "count": len(details),
        "faithfulness": avg_score("faithfulness"),
        "answer_relevancy": avg_score("answer_relevancy"),
        "answer_correctness": avg_score("answer_correctness"),
        "pass_rate": round(
            sum(1 for d in details if d.get("judge", {}).get("verdict") == "pass") / len(details), 4
        ) if details else 0.0,
    }

    report = {"summary": summary, "details": details}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"📄 Gemini judge report saved: {output_path}")


if __name__ == "__main__":
    main()
