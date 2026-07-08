# -*- coding: utf-8 -*-
"""
确定性 Guardrails 评估（离线，无 LLM Judge / 无 LangSmith）

评估对象已从旧的 Review ReAct Agent 改为确定性 guard 节点：
    workflows.legal_graph.make_guard_node(GuardrailsPipeline())
内部即 utils.guardrails.GuardrailsPipeline（PIIGuard→QualityGuard→CitationGuard→DisclaimerGuard）。

本脚本本地直跑，零外部服务（不连 LangSmith、不起 WorkflowService），可在 CI 离线运行。

评估指标：
- Guard Accuracy：期望被触发的 guard 集合 ∩ 实际触发集合 / 期望集合
- Status Accuracy：approve/revise 判断是否正确（revise = CitationGuard 触发重生成）
- Completion Rate：是否产出最终结果（final_answer 或 revise 标记非空即完成）

用法：
    python tests/eval_review_agent.py
    python tests/eval_review_agent.py --limit 5
"""
import sys
import asyncio
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from workflows.legal_graph import make_guard_node
from utils.guardrails import GuardrailsPipeline


# ==================== 旧工具名 → 确定性 guard 名映射 ====================
#
# 旧的 Review Agent 用 tools_called 表达期望（citation_check/add_disclaimer/
# pii_check/quality_check）；确定性管线里对齐为各 Guard 的 guard_name 中文名。

TOOL_TO_GUARD = {
    "pii_check": "PII脱敏",
    "citation_check": "引用验证",
    "add_disclaimer": "免责声明",
    "quality_check": "质量检查",
}


def _map_tools_to_guards(tools):
    """把旧 expected 里的 tools_called 映射成确定性 guard 名集合。"""
    return {TOOL_TO_GUARD[t] for t in tools if t in TOOL_TO_GUARD}


# ==================== 测试数据集 ====================
#
# 沿用旧的 REVIEW_DATASET（approve / revise / PII / 引用编造 等好场景），
# expected.tools_called 仍写旧工具名，评估时按 TOOL_TO_GUARD 映射成 guard 名。

REVIEW_DATASET = [
    # --- 应该 approve 的案例 ---
    {
        "input": {
            "draft": "根据《劳动法》第三十六条，劳动者每日工作时间不超过八小时。",
            "law_context": "第三十六条　国家实行劳动者每日工作时间不超过八小时、平均每周工作时间不超过四十四小时的工时制度。",
        },
        "expected": {
            "tools_called": ["citation_check", "add_disclaimer"],
            "review_status": "approve",
        },
    },
    {
        "input": {
            "draft": "您好，今天天气不错。",
            "law_context": "",
        },
        "expected": {
            "tools_called": [],
            "review_status": "approve",
        },
    },
    {
        "input": {
            "draft": "劳动合同应当以书面形式订立，这是基本要求。",
            "law_context": "第十九条　劳动合同应当以书面形式订立。",
        },
        "expected": {
            "tools_called": ["citation_check", "add_disclaimer"],
            "review_status": "approve",
        },
    },
    # --- 应该 revise 的案例（引用了不存在的法条） ---
    {
        "input": {
            "draft": "根据《劳动法》第一百二十条，用人单位可以随意解除劳动合同。",
            "law_context": "第三十六条　国家实行劳动者每日工作时间不超过八小时。",
        },
        "expected": {
            "tools_called": ["citation_check"],
            "review_status": "revise",
        },
    },
    {
        "input": {
            "draft": "根据《劳动合同法》第五十条和《劳动法》第一百条，用人单位必须缴纳社保。",
            "law_context": "第七十二条　社会保险基金按照保险类型确定资金来源。",
        },
        "expected": {
            "tools_called": ["citation_check", "add_disclaimer"],
            "review_status": "revise",
        },
    },
    # --- PII 检测案例 ---
    {
        "input": {
            "draft": "请联系张三，手机号13800138000，邮箱zhangsan@test.com。",
            "law_context": "",
        },
        "expected": {
            "tools_called": ["pii_check"],
            "review_status": "approve",
        },
    },
    {
        "input": {
            "draft": "员工身份证号为110101199001011234，银行卡号6222021234567890123。",
            "law_context": "",
        },
        "expected": {
            "tools_called": ["pii_check"],
            "review_status": "approve",
        },
    },
    # --- 质量检查案例 ---
    {
        "input": {
            "draft": "是的。",
            "law_context": "",
        },
        "expected": {
            "tools_called": ["quality_check"],
            "review_status": "approve",
        },
    },
    # --- 综合案例：PII + 引用问题 ---
    {
        "input": {
            "draft": "根据《劳动法》第二百条，联系李四13912345678了解详情。",
            "law_context": "第三十六条　国家实行劳动者每日工作时间不超过八小时。",
        },
        "expected": {
            "tools_called": ["pii_check", "citation_check"],
            "review_status": "revise",
        },
    },
    # --- 法律内容需要免责声明 ---
    {
        "input": {
            "draft": "劳动者有权拒绝违章指挥，这是法律赋予的权利。",
            "law_context": "第五十六条　劳动者对用人单位管理人员违章指挥、强令冒险作业，有权拒绝执行。",
        },
        "expected": {
            "tools_called": ["citation_check", "add_disclaimer"],
            "review_status": "approve",
        },
    },
]


# ==================== 单条评估（本地直跑 guard 节点） ====================

async def _eval_one(guard, item: dict) -> dict:
    """
    对一条用例构造 state，调用确定性 guard 节点，解析实际 status / 触发的 guard。

    - revise 判定：guard 返回 dict 含非空 guard_issues ⇒ "revise"，否则 "approve"。
    - 触发 guard 判定：
        * approve 分支：guard 走完整 pipeline，从 tool_history 最后一条的
          guards_triggered 取（make_guard_node 在 guard 历史项里存了名单）。
        * revise 分支：触发的是 "引用验证"。
    - completion：final_answer 非空 或 走了 revise（guard_issues 非空）即视为完成。
    """
    state = {
        "draft_answer": item["input"]["draft"],
        "law_context": item["input"].get("law_context", ""),
        "query": "",
        "tool_history": [],
        "guard_issues": [],
        "guard_retry": 0,
    }
    result = await guard(state)

    # status
    guard_issues = result.get("guard_issues", []) or []
    is_revise = bool(guard_issues)
    actual_status = "revise" if is_revise else "approve"

    # 实际触发的 guard 集合
    if is_revise:
        actual_guards = {"引用验证"}
    else:
        actual_guards = set()
        history = result.get("tool_history", []) or []
        if history:
            actual_guards = set(history[-1].get("guards_triggered", []) or [])

    # completion：拿到 final_answer 或走了 revise 都算完成
    completed = bool(result.get("final_answer")) or is_revise

    return {
        "actual_status": actual_status,
        "actual_guards": actual_guards,
        "completed": completed,
    }


def _score_one(item: dict, outcome: dict) -> dict:
    """根据期望与实际结果计算单条命中情况。"""
    expected = item["expected"]
    expected_guards = _map_tools_to_guards(expected.get("tools_called", []))
    expected_status = expected.get("review_status", "approve")

    actual_guards = outcome["actual_guards"]
    actual_status = outcome["actual_status"]

    # guard 命中率：期望 ∩ 实际 / 期望；期望为空时，实际也为空记 1.0，否则 0.0
    if not expected_guards:
        guard_score = 1.0 if not actual_guards else 0.0
        hits = set()
    else:
        hits = expected_guards & actual_guards
        guard_score = len(hits) / len(expected_guards)

    status_score = 1.0 if actual_status == expected_status else 0.0
    completion_score = 1.0 if outcome["completed"] else 0.0

    return {
        "expected_guards": expected_guards,
        "actual_guards": actual_guards,
        "hits": hits,
        "guard_score": guard_score,
        "expected_status": expected_status,
        "actual_status": actual_status,
        "status_score": status_score,
        "completion_score": completion_score,
    }


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(description="确定性 Guardrails 评估（离线）")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    dataset = REVIEW_DATASET[:args.limit] if args.limit > 0 else REVIEW_DATASET

    print(f"\n{'='*60}")
    print(f"确定性 Guardrails 评估 — {len(dataset)} 条测试数据（离线，无 LLM Judge / 无 LangSmith）")
    print(f"指标: Guard Accuracy + Status Accuracy + Completion")
    print(f"{'='*60}\n")

    # 本地构造确定性 guard 节点（零外部服务）
    guard = make_guard_node(GuardrailsPipeline())

    guard_scores = []
    status_scores = []
    completion_scores = []

    for idx, item in enumerate(dataset, 1):
        outcome = asyncio.run(_eval_one(guard, item))
        s = _score_one(item, outcome)

        guard_scores.append(s["guard_score"])
        status_scores.append(s["status_score"])
        completion_scores.append(s["completion_score"])

        draft_preview = item["input"]["draft"][:30]
        print(f"[{idx}/{len(dataset)}] {draft_preview}...")
        print(
            f"    status: 期望={s['expected_status']} 实际={s['actual_status']} "
            f"({'✓' if s['status_score'] == 1.0 else '✗'})"
        )
        print(
            f"    guard : 期望={sorted(s['expected_guards'])} 实际={sorted(s['actual_guards'])} "
            f"命中={sorted(s['hits'])} (score={s['guard_score']:.2f})"
        )
        print(
            f"    完成  : {'✓' if s['completion_score'] == 1.0 else '✗'}"
        )

    n = len(dataset) or 1
    guard_accuracy = sum(guard_scores) / n
    status_accuracy = sum(status_scores) / n
    completion_rate = sum(completion_scores) / n

    print(f"\n{'='*60}")
    print(f"评估结果")
    print(f"{'='*60}")
    print(f"  guard_accuracy : {guard_accuracy:.4f}")
    print(f"  status_accuracy: {status_accuracy:.4f}")
    print(f"  completion_rate: {completion_rate:.4f}")
    print(f"\n✅ 评估完成！")


if __name__ == "__main__":
    main()
