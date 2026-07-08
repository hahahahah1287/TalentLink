# -*- coding: utf-8 -*-
"""
统一法务图（确定性 DAG）

把原先两条图（contract_review 的 Plan-and-Execute + research_agent 的 ReAct）
收敛成**一条参数化的确定性流水线**：

    retrieve ──▶ run_skills ──▶ generate ──▶ guard ──▶ END
       │            │              │           │
   固化首节点    按路由确定性    LLM 只做合成   确定性 Guardrails
   (修 H3)      调 0~N skill   (无判定权)     (引用校验→可选 revise→
                (规则引擎/状态机/                PII→免责声明强制追加)
                 知识图谱/MMR)

设计要点（neuro-symbolic）：
- 把 LLM 关进笼子：判定（试用期超期？违规？时效届满？）交给 skill 的确定性内核，
  LLM 只负责"把结构化结论组织成通俗文字"。
- 编排是确定性 DAG，不靠 9B 自由调度（无 Planner / 无 ReAct / 无 recursion_limit）。
- 检索是图的首节点，所有法务问答都先查法条（修复"正常问答永不检索"的 H3）。
- 意图路由（utils/intent_router）是确定性关键词表，决定挂哪些 skill。
- guard 是确定性 GuardrailsPipeline，引用校验发现问题触发一次 revise（回 generate），
  PII 脱敏 / 免责声明强制后置，不交给 LLM 自选。

generate 节点同时支持 ainvoke（一次性，给 checkpoint 写入用）与
astream_text（逐 token，给 SSE 真流式用），见 workflow_service。
"""
import asyncio
import json
import time
from typing import Dict, Any, List, Callable, Optional

from langgraph.graph import StateGraph, END

from utils.state import AppState
from utils.retrieval_service import RetrievalService
from utils.tools.contract_tools import create_contract_chain, create_legal_qa_chain
from utils.guardrails import GuardrailsPipeline
from utils.evidence import extract_skill_evidence, render_skill_evidence


# ==================== skill 调度声明 ====================
#
# run_skills 节点按名字确定性调用 skill。每个 skill 声明：
#   - fn:  统一接口 fn(query, law_context="") -> str（结构化 JSON 文本）
#   - uses_law_context: 是否消费检索到的法条（修 M11：直调不能丢 law_context）
#   - label: 给 generate 的中文小标题
#
# 该表在 build_legal_graph 时由调用方注入（skill 已 init），这里只定义结构。

class SkillSpec:
    __slots__ = ("fn", "uses_law_context", "label")

    def __init__(self, fn: Callable[..., str], uses_law_context: bool, label: str):
        self.fn = fn
        self.uses_law_context = uses_law_context
        self.label = label


# ==================== 节点：检索（固化为首节点） ====================

def make_retrieve_node(retrieval_service: RetrievalService):
    """法条检索节点 —— 图的入口，永远先执行，不可被路由绕过。"""

    async def retrieve(state: AppState) -> Dict[str, Any]:
        query = state["query"]
        print(f"🔍 [Legal:retrieve] 检索法条: {query[:50]}...")
        # 同步检索（HyDE + 混合检索 + rerank）卸载到线程，避免阻塞事件循环
        if hasattr(retrieval_service, "retrieve_with_evidence"):
            retrieval = await asyncio.to_thread(
                retrieval_service.retrieve_with_evidence, query
            )
        else:
            # 兼容测试桩/旧检索器：无结构化证据时退回纯文本上下文
            law_context_legacy = await asyncio.to_thread(
                retrieval_service.retrieve_as_string, query
            )
            retrieval = {"docs": [], "evidence_items": [], "law_context": law_context_legacy}
        law_context = retrieval["law_context"]
        evidence_items = retrieval["evidence_items"]
        return {
            "retrieved_docs": retrieval["docs"],
            "evidence_items": evidence_items,
            "law_context": law_context,
            "tool_history": state.get("tool_history", []) + [{
                "step": "retrieve",
                "tool": "legal_retrieval",
                "input": query,
                "evidence_ids": [e.get("evidence_id") for e in evidence_items],
                "citations": [
                    (e.get("source") or {}).get("canonical_citation")
                    for e in evidence_items
                ],
                "output_len": len(law_context),
                "timestamp": time.time(),
            }],
        }

    return retrieve


# ==================== 节点：确定性 skill 执行 ====================

def make_run_skills_node(skill_specs: Dict[str, SkillSpec]):
    """
    按路由结果（state["route_skills"]）确定性地调用 0~N 个 skill。

    - 合同场景下，run_skills 的输入优先用 contract_text（风险条款抽取要看合同全文），
      其余 skill 用 query。
    - 消费法条的 skill 把 law_context 透传进去（修 M11）。
    - 每个 skill 用 to_thread 卸载（内部可能有 LLM 抽取/embedding），失败隔离不影响其它。
    """

    async def run_skills(state: AppState) -> Dict[str, Any]:
        skills: List[str] = state.get("route_skills", []) or []
        if not skills:
            return {"skill_outputs": state.get("skill_outputs", {})}

        query = state["query"]
        contract_text = state.get("contract_text") or ""
        law_context = state.get("law_context", "")
        has_contract = state.get("has_contract", False)

        async def _run_one(name: str):
            spec = skill_specs.get(name)
            if spec is None:
                return name, None
            # 风险条款识别要吃合同全文；其余默认用 query
            if name == "risk_clause_detector" and has_contract and contract_text:
                skill_input = contract_text
            elif has_contract and contract_text and name == "compliance_check":
                # 合同合规检查同样基于合同内容
                skill_input = contract_text
            else:
                skill_input = query
            lc = law_context if spec.uses_law_context else ""
            try:
                out = await asyncio.to_thread(spec.fn, skill_input, lc)
                return name, out
            except Exception as e:  # 单个 skill 失败隔离
                print(f"⚠️ [Legal:run_skills] skill {name} 执行失败: {e}")
                return name, json.dumps({"error": str(e), "skill": name}, ensure_ascii=False)

        results = await asyncio.gather(*[_run_one(s) for s in skills])

        skill_outputs = dict(state.get("skill_outputs", {}))
        history_entries = []
        for name, out in results:
            if out is None:
                continue
            skill_outputs[name] = out
            history_entries.append({
                "step": f"skill_{name}",
                "tool": name,
                "output_len": len(out),
                "timestamp": time.time(),
            })
            print(f"⚡ [Legal:run_skills] {name} 完成（{len(out)} 字）")

        return {
            "skill_outputs": skill_outputs,
            "tool_history": state.get("tool_history", []) + history_entries,
        }

    return run_skills


# ==================== skill 输出 → generate 上下文 ====================

def _format_skill_findings(state: AppState, skill_specs: Dict[str, SkillSpec]) -> str:
    """把 skill 的结构化 JSON 输出渲染成给合成 LLM 的【已识别要点】文本。"""
    outputs = state.get("skill_outputs", {})
    if not outputs:
        return "（无确定性分析结果）"
    parts = []
    skill_evidence = []
    for name, raw in outputs.items():
        label = skill_specs[name].label if name in skill_specs else name
        parts.append(f"【{label}（确定性分析结果）】\n{raw}")
        skill_evidence.extend(extract_skill_evidence(name, raw))
    evidence_text = render_skill_evidence(skill_evidence)
    if evidence_text:
        parts.append(evidence_text)
    return "\n\n".join(parts)


# ==================== 节点：合成（LLM 只做组织语言） ====================

def make_generate_node(contract_chain, legal_qa_chain, skill_specs: Dict[str, SkillSpec]):
    """
    合成节点。根据 has_contract 选择合同链 / 咨询链。

    既支持作为图节点被 ainvoke（写 final_answer，供 checkpoint），
    实际 SSE 真流式在 workflow_service 里直接调 build_generate_inputs + 链.astream。
    """

    async def generate(state: AppState) -> Dict[str, Any]:
        inputs, chain = build_generate_inputs(state, contract_chain, legal_qa_chain, skill_specs)
        is_retry = bool(state.get("guard_issues"))
        if is_retry:
            print("🔄 [Legal:generate] 引用校验反馈问题，重新合成...")
        else:
            print("📝 [Legal:generate] 合成最终回答...")
        result = await chain.ainvoke(inputs)
        return {
            "draft_answer": result,
            "final_answer": result,
            "guard_issues": [],  # 消费完清空
            "tool_history": state.get("tool_history", []) + [{
                "step": "generate_retry" if is_retry else "generate",
                "tool": "synthesis_chain",
                "output_len": len(result),
                "timestamp": time.time(),
            }],
        }

    return generate


def build_generate_inputs(state, contract_chain, legal_qa_chain, skill_specs):
    """
    构造合成链输入 + 选择链。抽出来供 generate 节点和 SSE astream 复用，
    保证"流式输出"和"一次性输出"走的是同一套 prompt/上下文。

    重试（guard 发现引用问题）时，在问题里追加纠偏指令。
    """
    skill_findings = _format_skill_findings(state, skill_specs)
    question = state["query"]
    guard_issues = state.get("guard_issues", [])
    if guard_issues:
        issues_text = "\n".join(f"- {i}" for i in guard_issues)
        allowed = "、".join(
            (e.get("source") or {}).get("canonical_citation", "")
            for e in state.get("evidence_items", [])
            if (e.get("source") or {}).get("canonical_citation")
        )
        question += (
            "\n\n【上次回答存在引用问题，请修正】\n" + issues_text +
            "\n请只引用【法律法规参考】中以【E编号】列出的证据，不要编造或引用未出现的条文号。"
            + (f"\n本轮允许引用的检索证据：{allowed}" if allowed else "")
        )

    law = state.get("law_context", "") or "未检索到相关法律条文"
    history = state.get("history", "")

    if state.get("has_contract"):
        inputs = {
            "history": history,
            "law": law,
            "contract": state.get("contract_text", "") or "（用户未单独提供合同文本，问题中已含合同内容）",
            "skill_findings": skill_findings,
            "question": question,
        }
        return inputs, contract_chain
    else:
        inputs = {
            "history": history,
            "law": law,
            "skill_findings": skill_findings,
            "question": question,
        }
        return inputs, legal_qa_chain


# ==================== 节点：确定性 Guardrails ====================

def make_guard_node(pipeline: Optional[GuardrailsPipeline] = None):
    """
    确定性输出防护节点（替代 Review ReAct Agent）。

    复用现成的 GuardrailsPipeline：PII 脱敏 → 质量检查 → 引用验证 → 免责声明。
    - 引用验证若发现编造法条，且尚未重生成过 → 标记 guard_issues 触发一次 revise（回 generate）。
    - PII / 免责声明等"追加型"防护强制后置执行，结果即 final_answer。

    与旧 Review Agent 的区别：纯确定性管线，不再让 9B 自主决定调哪些 guard、
    不再有 recursion_limit、不再需要 XML 清理。
    """
    pipeline = pipeline or GuardrailsPipeline()

    async def guard(state: AppState) -> Dict[str, Any]:
        draft = state.get("draft_answer", "") or state.get("final_answer", "")
        if not draft:
            return {"final_answer": "", "guard_issues": []}

        law_context = state.get("law_context", "")
        evidence_items = state.get("evidence_items", []) or []
        guard_retry = state.get("guard_retry", 0)
        guard_context = {"law_context": law_context, "evidence_items": evidence_items}

        # 引用验证单独先跑一次：决定是否需要 revise（在脱敏/追加免责之前判断更干净）
        from utils.guardrails import CitationGuard
        citation_result = CitationGuard().process(draft, context=guard_context)
        need_revise = (
            citation_result.modified
            and guard_retry < 1
            and law_context and len(law_context.strip()) >= 10
        )

        if need_revise:
            print(f"⚠️ [Legal:guard] 引用验证发现疑似编造法条，触发一次重生成: {citation_result.details}")
            return {
                "guard_issues": [citation_result.details],
                "guard_retry": guard_retry + 1,
                "tool_history": state.get("tool_history", []) + [{
                    "step": "guard_revise",
                    "tool": "citation_guard",
                    "details": citation_result.details,
                    "timestamp": time.time(),
                }],
            }

        # 不需要重生成 → 跑完整管线（含 PII 脱敏 + 强制免责声明）
        result = pipeline.run(draft, context=guard_context)
        final = result["output"]
        print(f"✅ [Legal:guard] 输出防护完成，触发: {[g['guard'] for g in result['guards_triggered']]}")
        return {
            "final_answer": final,
            "draft_answer": final,
            "guard_issues": [],
            "tool_history": state.get("tool_history", []) + [{
                "step": "guard",
                "tool": "guardrails_pipeline",
                "guards_triggered": [g["guard"] for g in result["guards_triggered"]],
                "output_len": len(final),
                "timestamp": time.time(),
            }],
        }

    return guard


def check_guard_result(state: AppState) -> str:
    """条件边：guard 发现引用问题 → 回 generate 重生成；否则结束。"""
    if state.get("guard_issues"):
        return "revise"
    return "done"


# ==================== 图构建 ====================

def build_legal_graph(
    llm,
    retrieval_service: RetrievalService,
    skill_specs: Dict[str, SkillSpec],
    guardrails: Optional[GuardrailsPipeline] = None,
):
    """
    构建统一法务图。

    流程：retrieve → run_skills → generate → guard → END
    guard 发现引用问题（首次）：guard → generate（带纠偏）→ guard → END

    Args:
        llm: LangChain LLM 实例（仅用于 generate 合成）
        retrieval_service: 统一检索服务
        skill_specs: {skill_name: SkillSpec}，由 WorkflowService 注入（skill 已初始化）
        guardrails: 可选，复用的 GuardrailsPipeline 实例

    Returns:
        编译好的 LangGraph graph
    """
    contract_chain = create_contract_chain(llm)
    legal_qa_chain = create_legal_qa_chain(llm)

    retrieve_node = make_retrieve_node(retrieval_service)
    run_skills_node = make_run_skills_node(skill_specs)
    generate_node = make_generate_node(contract_chain, legal_qa_chain, skill_specs)
    guard_node = make_guard_node(guardrails)

    workflow = StateGraph(AppState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("run_skills", run_skills_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("guard", guard_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "run_skills")
    workflow.add_edge("run_skills", "generate")
    workflow.add_edge("generate", "guard")
    workflow.add_conditional_edges(
        "guard",
        check_guard_result,
        {
            "revise": "generate",
            "done": END,
        },
    )

    return workflow.compile()
