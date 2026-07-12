# -*- coding: utf-8 -*-
"""
Skills 确定性内核单元测试

只测试不依赖 LLM / Embedding 的纯逻辑：
- 风险条款规则引擎 (risk_clause_detector.evaluate_rules / regex_extract)
- 合规决策表引擎 (compliance_check.evaluate_compliance / _apply_op)
- 时效状态机 (statute_checker.run_statute_machine)
- 术语知识图谱实体链接 (legal_term_explainer.link_entities / expand_related)
- 案例检索向量数学 + MMR (case_retriever.mmr_select / _cosine_sim_matrix)

LLM / Embedding 用桩对象（fake）隔离，确保测试快速、确定、可复现。

运行：python tests/test_skills.py
"""
import os
import sys
import json
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


# ==================== 轻量断言框架（不依赖 pytest） ====================

_PASS = 0
_FAIL = 0
_FAILURES = []


def check(name, cond, detail=""):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  ✅ {name}")
    else:
        _FAIL += 1
        _FAILURES.append(f"{name} — {detail}")
        print(f"  ❌ {name}  {detail}")


# ==================== 1. 风险条款规则引擎 ====================

def test_risk_rules():
    print("\n[1] 风险条款 规则引擎")
    from skills.risk_clause_detector import evaluate_rules, regex_extract

    # 试用期超期（合同12个月，试用6个月，分档上限2个月）→ 高风险
    r = evaluate_rules({"probation_months": 6, "contract_months": 12})
    types = [x["risk_type"] for x in r["risk_clauses"]]
    check("试用期6个月+合同12个月触发高风险",
          any("试用期" in t and x["risk_level"] == "高"
              for t, x in zip(types, r["risk_clauses"])),
          f"got={types}")
    check("汇总 high>=1", r["summary"]["high"] >= 1, f"summary={r['summary']}")

    # 合法试用期（合同36个月，试用2个月）→ 无风险
    r2 = evaluate_rules({"probation_months": 2, "contract_months": 36})
    check("合法试用期不报风险", len(r2["risk_clauses"]) == 0, f"got={r2['risk_clauses']}")

    # 违约金非服务期/竞业 → 高风险
    r3 = evaluate_rules({"has_penalty": True, "penalty_reason": "提前离职", "penalty_amount": 100000})
    check("违法违约金触发", any("违法约定违约金" in x["risk_type"] for x in r3["risk_clauses"]))

    # 违约金属于服务期 → 不报
    r4 = evaluate_rules({"has_penalty": True, "penalty_reason": "服务期培训"})
    check("服务期违约金不报", not any("违法约定违约金" in x["risk_type"] for x in r4["risk_clauses"]))

    # 竞业3年无补偿 → 中风险
    r5 = evaluate_rules({"noncompete_years": 3, "noncompete_compensation": False})
    check("竞业超期+无补偿触发", any("竞业限制" in x["risk_type"] for x in r5["risk_clauses"]))

    # 放弃社保 + 随意解除 + 不付加班费 → 3条高风险
    r6 = evaluate_rules({
        "waive_social_insurance": True,
        "arbitrary_termination": True,
        "overtime_unpaid": True,
    })
    check("三项高风险全部命中", r6["summary"]["high"] == 3, f"summary={r6['summary']}")

    # 每条风险都带法律依据（可解释性）
    all_have_basis = all(x.get("legal_basis") for x in r6["risk_clauses"])
    check("每条风险可追溯法条", all_have_basis)

    # 正则抽取
    facts = regex_extract("试用期为6个月，劳动合同期限为1年。违约金10万元。员工自愿加班不支付加班费。")
    check("正则抽取试用期=6", facts.get("probation_months") == 6, f"got={facts}")
    check("正则抽取合同=12个月", facts.get("contract_months") == 12, f"got={facts}")
    check("正则抽取违约金标记", facts.get("has_penalty") is True, f"got={facts}")
    check("正则抽取免加班费", facts.get("overtime_unpaid") is True, f"got={facts}")


# ==================== 2. 合规决策表引擎 ====================

def test_compliance_engine():
    print("\n[2] 合规检查 决策表引擎")
    from skills.compliance_check import evaluate_compliance, _apply_op, load_rules

    # 算子单测
    check("op gt", _apply_op(6, "gt", 5) is True)
    check("op lt", _apply_op(0.7, "lt", 0.8) is True)
    check("op false", _apply_op(False, "false", None) is True)
    check("op None 不命中", _apply_op(None, "gt", 5) is False)

    rules = load_rules()
    check("决策表非空", len(rules) > 0, f"rules={len(rules)}")

    # 试用期6个月 + 工资70% → 命中工资规则；试用期超期由 risk_clause 分档规则负责
    facts = {"probation_months": 6, "probation_wage_ratio": 0.7}
    res = evaluate_compliance(facts, rules)
    rule_ids = [v["rule_id"] for v in res["violations"]]
    check("试用期工资低命中 R-PROB-02", "R-PROB-02" in rule_ids, f"got={rule_ids}")
    # 绝对上限规则：试用期最长6个月合法，7个月才超绝对上限触发 R-PROB-01
    res_abs = evaluate_compliance({"probation_months": 7}, rules)
    check("试用期7个月命中 R-PROB-01",
          "R-PROB-01" in [v["rule_id"] for v in res_abs["violations"]],
          f"got={[v['rule_id'] for v in res_abs['violations']]}")

    # 每月加班80小时 + 不付加班费 → 命中
    facts2 = {"monthly_overtime_hours": 80, "overtime_pay_provided": False}
    res2 = evaluate_compliance(facts2, rules)
    ids2 = [v["rule_id"] for v in res2["violations"]]
    check("超时加班命中 R-OT-02", "R-OT-02" in ids2, f"got={ids2}")
    check("未付加班费命中 R-OT-03", "R-OT-03" in ids2, f"got={ids2}")

    # 合规场景（试用期1个月，工资90%）→ 无违规
    facts3 = {"probation_months": 1, "probation_wage_ratio": 0.9}
    res3 = evaluate_compliance(facts3, rules)
    check("合规场景无违规", len(res3["violations"]) == 0, f"got={res3['violations']}")

    # 覆盖率有意义（评估了字段才计入）
    check("覆盖率在(0,1]", 0 < res["coverage"] <= 1, f"coverage={res['coverage']}")

    # 违规项可追溯法条
    check("违规可追溯法条", all(v.get("legal_basis") for v in res["violations"]))


# ==================== 3. 时效状态机 ====================

def test_statute_machine():
    print("\n[3] 时效计算器 状态机")
    from skills.statute_checker import run_statute_machine, TimelineEvent

    # 基础：2024-03-01 起算，1年时效，基准日 2024-09-01 → 未届满
    res = run_statute_machine(date(2024, 3, 1), 365, [], as_of=date(2024, 9, 1))
    check("基础未届满", res.is_expired is False, f"remaining={res.days_remaining}")
    check("基础届满日正确", res.deadline == date(2025, 3, 1), f"deadline={res.deadline}")

    # 已届满：起算 2022-01-01，基准日 2024-09-01
    res2 = run_statute_machine(date(2022, 1, 1), 365, [], as_of=date(2024, 9, 1))
    check("已届满判定", res2.is_expired is True, f"remaining={res2.days_remaining}")

    # 中断：2024-03-01 起算，2024-09-01 中断 → 从 09-01 重新起算，届满 2025-09-01
    interrupt = [TimelineEvent("interrupt", date(2024, 9, 1), "协商")]
    res3 = run_statute_machine(date(2024, 3, 1), 365, interrupt, as_of=date(2024, 10, 1))
    check("中断后重新起算", res3.start_date == date(2024, 9, 1), f"start={res3.start_date}")
    check("中断后届满日顺延", res3.deadline == date(2025, 9, 1), f"deadline={res3.deadline}")
    check("中断次数=1", res3.interrupted_times == 1, f"times={res3.interrupted_times}")

    # 中止：起算 2024-01-01，中止 2024-02-01~2024-04-01(60天)，届满日顺延60天
    suspend = [
        TimelineEvent("suspend_start", date(2024, 2, 1)),
        TimelineEvent("suspend_end", date(2024, 4, 1)),
    ]
    res4 = run_statute_machine(date(2024, 1, 1), 365, suspend, as_of=date(2024, 5, 1))
    check("中止累计天数=60", res4.days_suspended == 60, f"suspended={res4.days_suspended}")
    expected_deadline = date(2024, 1, 1) + __import__("datetime").timedelta(days=365 + 60)
    check("中止后届满顺延", res4.deadline == expected_deadline, f"deadline={res4.deadline}")

    # 状态轨迹可解释
    check("有状态转移轨迹", len(res4.transitions) >= 2, f"transitions={res4.transitions}")


# ==================== 4. 术语知识图谱 ====================

def test_term_graph():
    print("\n[4] 术语解释 知识图谱实体链接")
    from skills.legal_term_explainer import link_entities, expand_related, build_knowledge_cards

    # 实体链接 + 别名
    linked = link_entities("我想了解经济补偿金和2N是什么")
    check("链接到经济补偿金", "经济补偿金" in linked, f"got={linked}")
    check("别名2N链接到赔偿金", "赔偿金" in linked, f"got={linked}")

    # 长词优先（"竞业限制"不被"竞业"截断成错误节点）
    linked2 = link_entities("竞业限制条款")
    check("竞业限制正确链接", "竞业限制" in linked2, f"got={linked2}")

    # 关系扩展：经济补偿金 → 易混的赔偿金
    related = expand_related("经济补偿金")
    rel_terms = [r["term"] for r in related]
    check("经济补偿金扩展出赔偿金", "赔偿金" in rel_terms, f"got={rel_terms}")
    check("关系边带类型", all(r.get("rel") for r in related))

    # 知识卡片含法条（grounding 锚点）
    cards = build_knowledge_cards("经济补偿金")
    check("卡片含法条", len(cards) > 0 and len(cards[0]["articles"]) > 0,
          f"cards={cards}")

    # 未知术语不误报
    linked3 = link_entities("今天天气不错")
    check("无关文本不链接", len(linked3) == 0, f"got={linked3}")


# ==================== 5. 案例检索 向量数学 + MMR ====================

def test_case_retrieval_math():
    print("\n[5] 相似案例检索 向量数学 + MMR")
    from skills.case_retriever import (
        _cosine_sim_matrix, _pairwise_cosine, mmr_select, _infer_categories, _element_boost
    )

    # 余弦相似度：与自身=1
    q = np.array([1.0, 0.0, 0.0])
    docs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    sims = _cosine_sim_matrix(q, docs)
    check("余弦自身≈1", abs(sims[0] - 1.0) < 1e-5, f"got={sims[0]}")
    check("余弦正交≈0", abs(sims[1]) < 1e-5, f"got={sims[1]}")
    check("余弦45度≈0.707", abs(sims[2] - 0.7071) < 1e-3, f"got={sims[2]}")

    # MMR：高相关但冗余的应被多样性惩罚
    # 候选0和1几乎相同(高相似)，候选2不同。relevance: [0.9,0.85,0.6]
    relevance = np.array([0.9, 0.85, 0.6])
    sim_matrix = np.array([
        [1.0, 0.98, 0.1],
        [0.98, 1.0, 0.1],
        [0.1, 0.1, 1.0],
    ])
    selected = mmr_select(relevance, sim_matrix, top_k=2, lambda_param=0.5)
    check("MMR首选最高相关(idx0)", selected[0] == 0, f"got={selected}")
    check("MMR次选多样的idx2而非冗余idx1", selected[1] == 2, f"got={selected}")

    # lambda=1 退化为纯相关性排序
    sel_pure = mmr_select(relevance, sim_matrix, top_k=2, lambda_param=1.0)
    check("lambda=1纯相关性选idx0,1", sel_pure == [0, 1], f"got={sel_pure}")

    # 要素推断 + 加成
    cats = _infer_categories("试用期6个月合法吗")
    check("推断出试用期类别", "试用期" in cats, f"got={cats}")
    boost = _element_boost("试用期超期问题", {"category": "试用期", "elements": {"scenario": "试用期超期"}})
    check("同类案例要素加成>0", boost > 0, f"boost={boost}")


# ==================== 6. 案例库 / 数据资产完整性 ====================

def test_data_assets():
    print("\n[6] 数据资产完整性 (案例库/决策表/术语图谱法条对齐)")
    import re

    base = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(base)

    # 知识库实际存在的法条号
    with open(os.path.join(root, "labor_law.txt"), encoding="utf-8") as f:
        law_text = f.read()
    cn_map = {'零':0,'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10,'百':100}
    def cn2int(s):
        result, temp = 0, 0
        for ch in s:
            v = cn_map.get(ch)
            if v is None: continue
            if v >= 10:
                temp = temp or 1
                result += temp * v; temp = 0
            else:
                temp = v
        return result + temp
    existing = set()
    for m in re.findall(r'第([一二三四五六七八九十百零]+)条', law_text):
        existing.add(cn2int(m))

    # 案例库法条都在知识库内
    with open(os.path.join(root, "skills/data/labor_cases.json"), encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    bad_case_articles = []
    for c in cases:
        for a in c.get("articles", []):
            if a not in existing:
                bad_case_articles.append((c["id"], a))
    check("案例库法条全部在《劳动法》内", not bad_case_articles, f"越界={bad_case_articles}")
    check("案例库>=10条", len(cases) >= 10, f"got={len(cases)}")

    # 术语图谱法条都在知识库内
    with open(os.path.join(root, "skills/data/legal_terms_graph.json"), encoding="utf-8") as f:
        terms = json.load(f)["terms"]
    bad_term_articles = []
    for name, info in terms.items():
        for a in info.get("articles", []):
            if a not in existing:
                bad_term_articles.append((name, a))
    check("术语图谱法条全部在《劳动法》内", not bad_term_articles, f"越界={bad_term_articles}")

    # 决策表结构完整
    with open(os.path.join(root, "skills/data/compliance_rules.json"), encoding="utf-8") as f:
        rules = json.load(f)["rules"]
    fields_ok = all(r.get("id") and r.get("op") and r.get("legal_basis") for r in rules)
    check("决策表每条规则字段完整", fields_ok)


# ==================== 7. 意图路由（确定性关键词 + 合同特征） ====================

def test_intent_router():
    print("\n[7] 意图路由 确定性关键词 + 合同内容特征")
    from utils.intent_router import (
        route_intent, looks_like_contract, contract_feature_score,
    )

    # --- 合同内容特征判定（不依赖前端字段） ---
    contract_doc = (
        "甲方（用人单位）与乙方（劳动者）经协商签订本劳动合同。"
        "第一条 劳动合同期限为3年。第二条 试用期为6个月，试用期工资为正式工资的70%。"
    )
    check("合同文本特征分超阈值", contract_feature_score(contract_doc) >= 4.0,
          f"score={contract_feature_score(contract_doc)}")
    check("识别为合同", looks_like_contract(contract_doc))
    check("普通咨询不是合同", not looks_like_contract("试用期最长能约定几个月？"))
    check("空文本不是合同", not looks_like_contract(""))
    check("过短文本不是合同(即便有甲乙方)", not looks_like_contract("甲方乙方"))

    # --- 合同场景：贴了合同进对话框（contract_text 为空，靠 query 内容判） ---
    r1 = route_intent(contract_doc)
    check("合同场景 has_contract=True", r1["has_contract"], f"{r1}")
    check("合同场景判定来源=content", r1["contract_source"] == "content", f"{r1}")
    check("合同场景必挂 risk_clause_detector", "risk_clause_detector" in r1["skills"], f"{r1}")
    check("合同场景必挂 compliance_check", "compliance_check" in r1["skills"], f"{r1}")

    # --- 合同走前端字段（query 是短问句，contract_text 才是合同） ---
    r2 = route_intent("帮我看看这份合同有什么风险", contract_text=contract_doc)
    check("前端字段也能判合同", r2["has_contract"], f"{r2}")
    check("字段来源标记正确", r2["contract_source"] in ("field", "both"), f"{r2}")

    # --- 时效咨询 → statute_checker ---
    r3 = route_intent("我去年被辞退了，现在申请劳动仲裁还来得及吗？")
    check("时效问题不判为合同", not r3["has_contract"], f"{r3}")
    check("时效问题路由到 statute_checker", "statute_checker" in r3["skills"], f"{r3}")

    # --- 术语解释 → legal_term_explainer ---
    r4 = route_intent("经济补偿金和赔偿金有什么区别？")
    check("术语问题路由到 legal_term_explainer",
          "legal_term_explainer" in r4["skills"], f"{r4}")

    # --- 相似案例 → case_retriever ---
    r5 = route_intent("我这种没签合同被辞退的情况，以前类似的案子是怎么判的？")
    check("案例问题路由到 case_retriever", "case_retriever" in r5["skills"], f"{r5}")

    # --- 合规咨询 → compliance_check ---
    r6 = route_intent("每天加班4小时不给加班费合法吗？")
    check("合规问题路由到 compliance_check", "compliance_check" in r6["skills"], f"{r6}")

    # --- 法务兜底：命中法务词但无具体 skill → compliance_check 兜底 ---
    r7 = route_intent("公司没给我交社保")
    check("法务兜底触发 compliance_check", "compliance_check" in r7["skills"], f"{r7}")

    # --- 纯闲聊 → 不判合同、无 skill（仅检索→合成） ---
    r8 = route_intent("你好，今天天气怎么样")
    check("闲聊不判为合同", not r8["has_contract"], f"{r8}")
    check("闲聊无 skill", r8["skills"] == [], f"{r8}")

    # --- skills 去重保序 ---
    r9 = route_intent(contract_doc + " 这种情况以前怎么判的？")
    check("合同+案例 skill 不重复", len(r9["skills"]) == len(set(r9["skills"])), f"{r9}")
    check("合同+案例同时挂 case_retriever", "case_retriever" in r9["skills"], f"{r9}")


# ==================== 8. 外部检索官方来源白名单 ====================

def test_web_search_evidence_policy():
    print("\n[8] 外部检索 官方来源白名单 / URL evidence")
    from skills.web_search import _is_trusted_official_domain, _normalize_result

    check("gov.cn 子域名可信", _is_trusted_official_domain("https://www.mohrss.gov.cn/xxgk2020/"))
    check("伪 gov.cn 后缀不可信", not _is_trusted_official_domain("https://www.mohrss.gov.cn.evil.example/a"))

    item = _normalize_result(
        {
            "title": "最低工资标准",
            "link": "https://www.mohrss.gov.cn/xxgk2020/fdzdgknr/zcfg/",
            "snippet": "官方发布摘要",
        },
        "最低工资标准",
        "2026-07-08T00:00:00+00:00",
    )
    check("官方结果标记 trusted", item["trusted"] is True, f"item={item}")
    check("外部证据带 URL", item["source_kind"] == "external_url" and bool(item["url"]), f"item={item}")
    check("外部证据带 content_hash", bool(item.get("content_hash")), f"item={item}")
    check("外部证据声明白名单版本", item.get("whitelist_version") == "official-cn-web-v1", f"item={item}")


# ==================== 9. 确定性 Guard 节点逻辑 ====================

def test_guard_logic():
    print("\n[9] 确定性 Guard 引用校验 / 免责声明 / revise 触发")
    import asyncio
    from utils.guardrails import GuardrailsPipeline, CitationGuard
    from workflows.legal_graph import make_guard_node, check_guard_result

    law_ctx = "《劳动法》第二十一条 劳动合同可以约定试用期。第四十四条 延长工作时间应支付加班费。"

    # 引用了检索结果里没有的条文（第99条）→ 首次应触发 revise
    draft_bad = "根据《劳动法》第99条，你可以要求三倍赔偿。"
    guard = make_guard_node(GuardrailsPipeline())
    state = {
        "draft_answer": draft_bad,
        "law_context": law_ctx,
        "guard_retry": 0,
    }
    out = asyncio.run(guard(state))
    check("编造法条→标记 guard_issues", bool(out.get("guard_issues")), f"{out}")
    check("编造法条→guard_retry+1", out.get("guard_retry") == 1, f"{out}")
    check("revise 分支生效", check_guard_result(out) == "revise", f"{out}")

    # 已经重生成过一次（guard_retry=1）→ 不再 revise，跑完整管线
    state2 = {
        "draft_answer": draft_bad,
        "law_context": law_ctx,
        "guard_retry": 1,
    }
    out2 = asyncio.run(guard(state2))
    check("二次不再 revise", not out2.get("guard_issues"), f"{out2}")
    check("二次产出 final_answer", bool(out2.get("final_answer")), f"{out2}")
    check("引用问题在终稿里被标注",
          "引用验证" in out2.get("final_answer", "") or "可能不准确" in out2.get("final_answer", ""),
          f"{out2.get('final_answer','')[:120]}")

    # 引用合法（第二十一条在检索结果里）→ 不 revise，免责声明强制追加
    draft_ok = "根据《劳动法》第二十一条，试用期最长不超过6个月。"
    state3 = {
        "draft_answer": draft_ok,
        "law_context": law_ctx,
        "guard_retry": 0,
    }
    out3 = asyncio.run(guard(state3))
    check("合法引用不触发 revise", not out3.get("guard_issues"), f"{out3}")
    check("法务回答强制追加免责声明", "免责声明" in out3.get("final_answer", ""),
          f"{out3.get('final_answer','')[-80:]}")
    check("done 分支生效", check_guard_result(out3) == "done", f"{out3}")

    # 无草稿 → 安全返回空
    out4 = asyncio.run(guard({"draft_answer": "", "law_context": law_ctx}))
    check("空草稿安全返回", out4.get("final_answer") == "", f"{out4}")


# ==================== 主流程 ====================

def main():
    print("=" * 60)
    print("Skills 确定性内核单元测试")
    print("=" * 60)

    test_risk_rules()
    test_compliance_engine()
    test_statute_machine()
    test_term_graph()
    test_case_retrieval_math()
    test_data_assets()
    test_intent_router()
    test_web_search_evidence_policy()
    test_guard_logic()

    print("\n" + "=" * 60)
    print(f"结果: {_PASS} 通过, {_FAIL} 失败")
    print("=" * 60)
    if _FAILURES:
        print("\n失败详情:")
        for f in _FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("✅ 全部通过")


if __name__ == "__main__":
    main()
