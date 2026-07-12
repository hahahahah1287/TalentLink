# -*- coding: utf-8 -*-
"""轻量图编译 + e2e 烟雾测试（stub，不加载 9B）。"""
import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflows.legal_graph import build_legal_graph, SkillSpec
from utils.guardrails import GuardrailsPipeline
from utils.skill_result import make_skill_result, parse_skill_result
import workflows.legal_graph as lg


class StubChain:
    async def ainvoke(self, inputs):
        return "STUB ANSWER"

    async def astream(self, inputs):
        for t in ["根据", "《劳动法》", "第二十一条", "，试用期最长6个月。"]:
            yield t


class StubLLM:
    pass


class StubRetrieval:
    def retrieve_as_string(self, q):
        return "《劳动法》第二十一条 劳动合同可以约定试用期，试用期最长不得超过六个月。"


def fake_skill(q, lc=""):
    return json.dumps(
        make_skill_result(
            skill_name="stub_skill",
            findings=[
                {"type": "valid", "message": "结构化 finding 已生成"},
                {"type": ""},
            ],
            display_text=f"input_len={len(q)}, used_law={bool(lc)}",
            metrics={"input_len": len(q), "used_law": bool(lc)},
        ),
        ensure_ascii=False,
    )


def main():
    lg.create_contract_chain = lambda llm: StubChain()
    lg.create_legal_qa_chain = lambda llm: StubChain()

    specs = {
        "compliance_check": SkillSpec(fake_skill, False, "合规检查"),
        "statute_checker": SkillSpec(fake_skill, False, "时效计算"),
    }
    g = build_legal_graph(StubLLM(), StubRetrieval(), specs, GuardrailsPipeline())
    print("graph compiled:", type(g).__name__)

    state = {
        "query": "试用期最长几个月？还能不能申请仲裁",
        "has_contract": False,
        "route_skills": ["compliance_check", "statute_checker"],
        "tool_history": [],
        "skill_outputs": {},
        "guard_issues": [],
        "guard_retry": 0,
        "history": "",
    }
    out = asyncio.run(g.ainvoke(state))
    fa = out.get("final_answer", "")
    print("final_answer:", fa[:120])
    print("skills ran:", list(out.get("skill_outputs", {}).keys()))
    print("law_context retrieved:", bool(out.get("law_context")))
    print("disclaimer appended:", "免责声明" in fa)
    assert fa, "final_answer is empty"
    assert out.get("skill_outputs"), "no skills ran"
    assert out.get("law_context"), "law_context not retrieved"

    cleaned = parse_skill_result(out["skill_outputs"]["compliance_check"])
    assert cleaned["skill_name"] == "compliance_check", "specialist name was not normalized"
    assert len(cleaned.get("findings", [])) == 1, "malformed finding was not dropped"
    assert out.get("specialist_reports"), "missing specialist reports"
    assert out.get("specialist_corrections"), "missing specialist correction audit"
    assert any(c.get("action") == "drop_finding" for c in out["specialist_corrections"]), "drop_finding correction not recorded"
    print("specialist corrections:", out.get("specialist_corrections"))
    print("--- graph e2e (stub) OK ---")


if __name__ == "__main__":
    main()
