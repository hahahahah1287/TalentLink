# -*- coding: utf-8 -*-
"""
全链路端到端评估脚本（不需要 LLM Judge）

评估**单一确定性法务图**的端到端表现：

    retrieve ──▶ run_skills ──▶ generate ──▶ guard ──▶ END

（旧的双图：contract_review 的 Plan-and-Execute + research_agent 的 ReAct，
  以及 Review ReAct Agent，已收敛为这一条参数化的确定性流水线。）

指标：
1. Pipeline Completion Rate — 工作流是否正常完成（产出有效 final_answer）
2. Tool Chain Coverage — tool_history 是否覆盖预期工具链（legal_retrieval /
   各 skill / synthesis_chain / guardrails_pipeline，coverage 允许部分命中）
3. Answer Relevance — 最终回答与标准答案的关键词重叠度（jieba 分词）
4. Latency — 端到端耗时

合同识别基于内容特征（utils.intent_router.route_intent），不依赖前端字段；
是否挂 skill、挂哪些 skill 同样由确定性关键词路由决定。

用法：
    python tests/eval_e2e.py
    python tests/eval_e2e.py --limit 3
    python tests/eval_e2e.py --output my_report.json
"""
import os
import sys
import json
import re
import time
import asyncio
import argparse
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).parent.parent))

import jieba

from config import AppConfig
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.retrievers import BM25Retriever
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever
try:
    from langchain_experimental.chat_models import ChatLlamaCpp
except ImportError:
    from langchain_community.chat_models import ChatLlamaCpp

from utils import RerankService, RetrievalService
from utils.guardrails import GuardrailsPipeline
from utils.intent_router import route_intent
from workflows import build_legal_graph, SkillSpec

# skill 实现 + 初始化（与 workflow_service.py 保持一致：
# risk/compliance/term 传 llm，case 传 embeddings，statute 无需 init）
from skills.risk_clause_detector import risk_clause_skill, init_skill as init_risk_skill
from skills.compliance_check import compliance_skill, init_skill as init_compliance_skill
from skills.legal_term_explainer import legal_term_skill, init_skill as init_term_skill
from skills.statute_checker import statute_skill
from skills.case_retriever import case_retriever_skill, init_skill as init_case_skill


# ==================== 测试数据集 ====================
#
# 合同审查用例（route_intent 会判定 has_contract=True，固定挂
# risk_clause_detector + compliance_check）。expected_tools 写成新 tool_history
# 里实际出现的 tool 名：legal_retrieval / <skill_name> / synthesis_chain /
# guardrails_pipeline。已删除 review 概念（无 expected_review_status）。

CONTRACT_EVAL_DATASET = [
    {
        "query": "这份合同的试用期约定是否合法？",
        "contract_text": "甲方与乙方约定：试用期为6个月，劳动合同期限为1年。试用期工资为正式工资的70%。",
        "ground_truth": "根据劳动法第十九条，劳动合同期限一年以上不满三年的，试用期不得超过二个月。本合同约定试用期6个月违反了法律规定。",
        "expected_tools": ["legal_retrieval", "risk_clause_detector", "compliance_check", "synthesis_chain", "guardrails_pipeline"],
    },
    {
        "query": "合同中关于加班费的条款是否符合劳动法？",
        "contract_text": "员工加班按照基本工资的1.5倍支付加班费，法定节假日加班按照2倍支付。",
        "ground_truth": "根据劳动法第四十四条，法定节假日加班应支付不低于工资的300%的报酬，周末加班应支付不低于200%的报酬。该合同法定节假日加班费2倍不符合规定。",
        "expected_tools": ["legal_retrieval", "risk_clause_detector", "compliance_check", "synthesis_chain", "guardrails_pipeline"],
    },
    {
        "query": "合同解除条款是否合法？",
        "contract_text": "甲方有权在任何时候以任何理由解除劳动合同，且无需支付经济补偿金。本合同自双方签订之日起生效。",
        "ground_truth": "根据劳动合同法，用人单位解除劳动合同需要符合法定情形，违法解除需支付赔偿金。该条款排除了劳动者的合法权益，属于无效条款。",
        "expected_tools": ["legal_retrieval", "risk_clause_detector", "compliance_check", "synthesis_chain", "guardrails_pipeline"],
    },
    {
        "query": "这份劳动合同中的竞业限制条款是否合理？",
        "contract_text": "甲方与乙方约定：乙方离职后2年内不得在同行业就业，竞业限制期间每月补偿500元。",
        "ground_truth": "根据劳动合同法第二十四条，竞业限制期限不得超过二年。竞业限制补偿金不得低于劳动者离职前十二个月平均工资的30%。500元的补偿金额可能过低。",
        "expected_tools": ["legal_retrieval", "risk_clause_detector", "compliance_check", "synthesis_chain", "guardrails_pipeline"],
    },
    {
        "query": "合同中的社会保险条款是否合规？",
        "contract_text": "甲方为乙方缴纳社会保险，个人应缴部分由甲方代扣代缴。乙方自愿放弃缴纳社会保险的，甲方每月补贴300元。",
        "ground_truth": "根据劳动法第七十二条，用人单位和劳动者必须依法参加社会保险，缴纳社会保险费。劳动者放弃社保的约定无效。",
        "expected_tools": ["legal_retrieval", "risk_clause_detector", "compliance_check", "synthesis_chain", "guardrails_pipeline"],
    },
    {
        "query": "合同中关于年假的约定是否合法？",
        "contract_text": "甲方与乙方约定：员工工作满一年后享受5天年假，工作满十年后享受10天年假。年假不可跨年累积。",
        "ground_truth": "根据职工带薪年休假条例，职工累计工作满1年不满10年的，年休假5天；满10年不满20年的，年休假10天；满20年的，年休假15天。年假安排是用人单位的义务。",
        "expected_tools": ["legal_retrieval", "risk_clause_detector", "compliance_check", "synthesis_chain", "guardrails_pipeline"],
    },
    {
        "query": "工资支付条款是否合规？",
        "contract_text": "甲方每月15日以银行转账方式支付上月工资。如遇节假日，提前至最近的工作日支付。本合同期限为3年。",
        "ground_truth": "根据劳动法第五十条，工资应当以货币形式按月支付给劳动者本人，不得克扣或者无故拖欠劳动者的工资。该条款约定合理合法。",
        "expected_tools": ["legal_retrieval", "risk_clause_detector", "compliance_check", "synthesis_chain", "guardrails_pipeline"],
    },
    {
        "query": "合同中的违约金条款是否合法？",
        "contract_text": "甲方与乙方约定：乙方提前解除劳动合同的，需向甲方支付违约金10万元。本合同期限为2年。",
        "ground_truth": "根据劳动合同法第二十五条，除服务期和竞业限制外，用人单位不得与劳动者约定由劳动者承担违约金。该违约金条款违法。",
        "expected_tools": ["legal_retrieval", "risk_clause_detector", "compliance_check", "synthesis_chain", "guardrails_pipeline"],
    },
    {
        "query": "工伤条款是否符合法律规定？",
        "contract_text": "甲方与乙方约定：员工在工作期间受伤，公司负责治疗费用。因员工自身原因导致的工伤，公司不承担责任。",
        "ground_truth": "根据工伤保险条例，工伤认定不以劳动者是否有过错为条件。即使是劳动者自身过失导致的伤害，只要符合工伤认定条件就应认定为工伤。",
        "expected_tools": ["legal_retrieval", "risk_clause_detector", "compliance_check", "synthesis_chain", "guardrails_pipeline"],
    },
    {
        "query": "劳动合同中关于培训费用的条款是否合法？",
        "contract_text": "甲方为乙方提供专项培训费用5万元，乙方需服务满5年。未满5年离职的，需全额退还培训费用。本合同期限为5年。",
        "ground_truth": "根据劳动合同法第二十二条，用人单位为劳动者提供专项培训费用可以约定服务期。违约金不得超过用人单位提供的培训费用，且应按已履行期限递减。",
        "expected_tools": ["legal_retrieval", "risk_clause_detector", "compliance_check", "synthesis_chain", "guardrails_pipeline"],
    },
]

# 普通法务咨询用例（route_intent 判定 has_contract=False，按关键词挑 skill）。
# 含合同特征的"贴合同进对话框"用例同样会被识别为 has_contract，挂风险+合规。
RESEARCH_EVAL_DATASET = [
    {
        "query": "劳动法第三十八条规定了什么内容？",
        "ground_truth": "劳动法第三十八条规定用人单位应当保证劳动者每周至少休息一日。",
        "expected_tools": ["legal_retrieval", "synthesis_chain", "guardrails_pipeline"],
    },
    {
        "query": "2024年最低工资标准是多少？",
        "ground_truth": "最低工资标准由各省级人民政府确定，各地标准不同，需要查询具体地区的最新标准。",
        "expected_tools": ["legal_retrieval", "compliance_check", "synthesis_chain", "guardrails_pipeline"],
    },
    {
        "query": "劳动合同法关于无固定期限劳动合同的签订条件是什么？",
        "ground_truth": "劳动合同法第十四条规定，劳动者在该用人单位连续工作满十年的，或者连续订立二次固定期限劳动合同的，应当订立无固定期限劳动合同。",
        "expected_tools": ["legal_retrieval", "legal_term_explainer", "synthesis_chain", "guardrails_pipeline"],
    },
    {
        "query": "什么是N+1补偿？劳动法是怎么规定的？",
        "ground_truth": "N+1补偿是指用人单位解除劳动合同时，按劳动者工作年限每满一年支付一个月工资的经济补偿，再额外支付一个月工资作为代通知金。",
        "expected_tools": ["legal_retrieval", "legal_term_explainer", "synthesis_chain", "guardrails_pipeline"],
    },
    {
        "query": "劳动仲裁的时效是多长？",
        "ground_truth": "根据劳动争议调解仲裁法第二十七条，劳动争议申请仲裁的时效期间为一年，从当事人知道或者应当知道其权利被侵害之日起计算。",
        "expected_tools": ["legal_retrieval", "statute_checker", "synthesis_chain", "guardrails_pipeline"],
    },
    # --- skill 用例：风险条款识别（query 含合同特征，会被判为 has_contract） ---
    {
        "query": "请帮我检查这份合同有没有风险条款：甲方与乙方约定试用期为8个月，劳动合同期限为2年。员工自愿加班，公司不支付加班费。员工离职后3年内不得在同行业就业。",
        "ground_truth": "该合同存在多个风险条款：试用期8个月超过法定上限2个月；约定不支付加班费违反劳动法；竞业限制3年超过法定上限2年。",
        "expected_tools": ["legal_retrieval", "risk_clause_detector", "compliance_check", "synthesis_chain", "guardrails_pipeline"],
    },
    {
        "query": "分析这份合同的风险：甲方有权随时解除劳动合同且无需补偿。乙方入职需缴纳押金2000元。违约金5万元。本合同期限为3年。",
        "ground_truth": "该合同存在严重风险：单方随时解除条款违法；收取押金违法；除服务期和竞业限制外不得约定违约金。",
        "expected_tools": ["legal_retrieval", "risk_clause_detector", "compliance_check", "synthesis_chain", "guardrails_pipeline"],
    },
    # --- skill 用例：合规检查 ---
    {
        "query": "公司规定试用期6个月，劳动合同签了1年，试用期工资是正式工资的70%，这样合法吗？",
        "ground_truth": "不合法。1年期限的劳动合同试用期不得超过1个月，试用期工资不得低于80%。",
        "expected_tools": ["legal_retrieval", "compliance_check", "synthesis_chain", "guardrails_pipeline"],
    },
    {
        "query": "我们公司要求员工每天加班4小时，没有加班费，只给调休，这样合规吗？",
        "ground_truth": "不合规。每日加班不超过1小时特殊情况不超过3小时，每月不超过36小时。加班必须支付加班费，不能仅以调休替代。",
        "expected_tools": ["legal_retrieval", "compliance_check", "synthesis_chain", "guardrails_pipeline"],
    },
    # --- skill 用例：法律术语解释 ---
    {
        "query": "帮我解释一下什么是经济补偿金和赔偿金，有什么区别？",
        "ground_truth": "经济补偿金是用人单位合法解除劳动合同时支付的补偿，按工作年限每满一年支付一个月工资。赔偿金是违法解除时支付的惩罚性赔偿，标准是经济补偿金的两倍。",
        "expected_tools": ["legal_retrieval", "legal_term_explainer", "synthesis_chain", "guardrails_pipeline"],
    },
    {
        "query": "合同里提到了竞业限制和服务期协议，我不太懂是什么意思，能解释一下吗？",
        "ground_truth": "竞业限制是离职后一定时间内不能去竞争对手公司工作，单位需按月支付补偿金。服务期协议是单位出钱培训后约定必须工作一定年限，提前离职需退还培训费。",
        "expected_tools": ["legal_retrieval", "legal_term_explainer", "synthesis_chain", "guardrails_pipeline"],
    },
    # --- skill 用例：时效计算器 ---
    {
        "query": "我2024年3月被公司违法辞退了，现在还能申请劳动仲裁吗？",
        "ground_truth": "劳动仲裁时效为1年，从知道权利被侵害之日起计算。2024年3月被辞退，仲裁时效至2025年3月届满。",
        "expected_tools": ["legal_retrieval", "statute_checker", "synthesis_chain", "guardrails_pipeline"],
    },
    {
        "query": "公司从2023年1月开始拖欠我的工资，我现在还能申请仲裁吗？过没过期？",
        "ground_truth": "劳动关系存续期间因拖欠劳动报酬发生争议的，不受1年仲裁时效限制。在职期间可以随时申请仲裁。",
        "expected_tools": ["legal_retrieval", "statute_checker", "synthesis_chain", "guardrails_pipeline"],
    },
]

# 统一喂给同一张法务图的合并用例集
ALL_EVAL_DATASET = CONTRACT_EVAL_DATASET + RESEARCH_EVAL_DATASET


# ==================== 评估指标 ====================

def _tokenize(text: str) -> set:
    """jieba 分词 + 法条引用 + 英文 + 数字"""
    article_refs = set(re.findall(r'第[一二三四五六七八九十百千零\d]+[条章节款项目]', text))
    stop_words = {'的', '了', '在', '是', '和', '与', '或', '及', '等',
                  '对', '为', '不', '有', '这', '那', '被', '从', '到',
                  '中', '上', '下', '内', '外', '其', '该', '将', '已',
                  '也', '都', '而', '但', '如', '则', '可', '应', '要',
                  '会', '能', '以', '于', '由', '因', '此', '之', '所',
                  '请', '需', '按', '每', '一', '个', '条', '款', '人',
                  '我', '你', '他', '她', '它', '们', '没', '很', '还'}
    words = set(w for w in jieba.lcut(text) if len(w) >= 2 and w not in stop_words)
    en_words = set(re.findall(r'[a-zA-Z]+', text.lower()))
    numbers = set(re.findall(r'\d+', text))
    return words | article_refs | en_words | numbers


def compute_answer_relevance(final_answer: str, ground_truth: str) -> float:
    """计算最终回答与标准答案的关键词重叠度"""
    if not final_answer or not ground_truth:
        return 0.0
    gt_tokens = _tokenize(ground_truth)
    ans_tokens = _tokenize(final_answer)
    if not gt_tokens:
        return 0.0
    return len(gt_tokens & ans_tokens) / len(gt_tokens)


def check_tool_chain(tool_history: list, expected_tools: list) -> dict:
    """检查工具链是否与预期匹配"""
    actual_tools = [t.get("tool", "") for t in tool_history]
    expected_seq = expected_tools[:]

    # 精确匹配序列
    exact_match = actual_tools == expected_seq

    # 包含匹配（不要求顺序）
    actual_set = set(actual_tools)
    expected_set = set(expected_seq)
    hit_count = len(actual_set & expected_set)
    coverage = hit_count / len(expected_set) if expected_set else 1.0

    return {
        "exact_match": exact_match,
        "coverage": coverage,
        "actual": actual_tools,
        "expected": expected_seq,
    }


def check_completion(state: dict) -> dict:
    """检查 pipeline 是否正常完成"""
    final_answer = state.get("final_answer", "")
    has_answer = bool(final_answer and len(final_answer.strip()) > 10)
    messages_count = len(state.get("tool_history", []))
    return {
        "completed": has_answer,
        "answer_len": len(final_answer),
        "steps": messages_count,
    }


# ==================== 构建轻量组件 ====================

def build_components(config: AppConfig):
    """构建最小组件集，不加载 MySQL/Redis。返回 (llm, retrieval_service, embeddings)。"""
    print("📦 加载 Embedding...")
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=config.embedding.model_name,
        model_kwargs={'device': config.embedding.device, 'local_files_only': True},
        encode_kwargs={'normalize_embeddings': config.embedding.normalize_embeddings}
    )

    print("📦 加载 Reranker...")
    reranker = RerankService(
        model_name=config.reranker.model_name,
        device=config.reranker.device,
        batch_size=config.reranker.batch_size
    )

    print("📦 加载向量库...")
    metadata_index_path = config.retrieval.faiss_index_path + "_with_metadata"
    vector_store = FAISS.load_local(
        metadata_index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    docs = list(vector_store.docstore._dict.values())

    faiss_ret = vector_store.as_retriever(search_kwargs={"k": config.retrieval.retrieval_k})
    bm25_ret = BM25Retriever.from_documents(docs)
    bm25_ret.k = config.retrieval.retrieval_k

    ensemble = EnsembleRetriever(
        retrievers=[bm25_ret, faiss_ret],
        weights=[config.retrieval.bm25_weight, config.retrieval.faiss_weight],
    )

    retrieval_service = RetrievalService(
        retriever=ensemble,
        reranker=reranker,
        top_k=config.reranker.top_k,
        rerank_enabled=config.retrieval.rerank_enabled,
        score_threshold=config.reranker.score_threshold,
        hyde_enabled=config.retrieval.hyde_enabled,
    )

    print(f"📦 加载 LLM: {config.llm.model_path}...")
    llm = ChatLlamaCpp(
        model_path=config.llm.model_path,
        n_gpu_layers=config.llm.n_gpu_layers,
        n_ctx=config.llm.n_ctx,
        temperature=config.llm.temperature,
        verbose=False,
        streaming=False,
    )
    print("✅ 所有组件加载完成")

    return llm, retrieval_service, embeddings


def build_graph(llm, retrieval_service, embeddings):
    """
    用已加载的组件构建统一法务图（与 workflow_service.py 的接线对齐）。

    init 5 个 skill：risk/compliance/term 传 llm，case 传 embeddings，
    statute 无需 init；统一接口 fn(query, law_context="")，uses_law_context 全 False。
    """
    # 初始化 skill（复用已加载的 llm / embeddings）
    init_risk_skill(llm)
    init_compliance_skill(llm)
    init_term_skill(llm)
    init_case_skill(embeddings)

    # 直接构造 skill_specs（{name: SkillSpec(fn, uses_law_context, label)}），
    # uses_law_context 全部 False，label 用中文（与 WorkflowService 注册一致）。
    skill_specs = {
        "risk_clause_detector": SkillSpec(risk_clause_skill, False, "合同风险条款"),
        "compliance_check": SkillSpec(compliance_skill, False, "合规检查"),
        "legal_term_explainer": SkillSpec(legal_term_skill, False, "术语解释"),
        "statute_checker": SkillSpec(statute_skill, False, "时效计算"),
        "case_retriever": SkillSpec(case_retriever_skill, False, "相似案例"),
    }

    return build_legal_graph(llm, retrieval_service, skill_specs, GuardrailsPipeline())


# ==================== 评估运行器 ====================

async def run_case(graph, case: dict) -> dict:
    """运行单条法务图测试用例（合同/咨询统一走同一张图）。"""
    # 用确定性意图路由填充 has_contract / route_skills，对齐新 AppState
    route = route_intent(case["query"], case.get("contract_text"))
    initial_state = {
        "query": case["query"],
        "contract_text": case.get("contract_text", ""),
        "history": "",
        "scene": "legal",
        "has_contract": route["has_contract"],
        "route_skills": route["skills"],
        "tool_history": [],
        "skill_outputs": {},
        "status": "running",
        "guard_issues": [],
        "guard_retry": 0,
    }

    start = time.time()
    try:
        result = await asyncio.wait_for(graph.ainvoke(initial_state), timeout=180)
        latency = time.time() - start
    except asyncio.TimeoutError:
        return {"error": "timeout", "latency": 180.0}
    except Exception as e:
        return {"error": str(e), "latency": time.time() - start}

    tool_check = check_tool_chain(result.get("tool_history", []), case["expected_tools"])
    completion = check_completion(result)
    relevance = compute_answer_relevance(result.get("final_answer", ""), case["ground_truth"])

    return {
        "final_answer": result.get("final_answer", "")[:500],
        "has_contract": route["has_contract"],
        "route_skills": route["skills"],
        "tool_chain": tool_check,
        "completion": completion,
        "answer_relevance": round(relevance, 4),
        "latency": round(latency, 2),
    }


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(description="统一法务图全链路端到端评估")
    parser.add_argument("--limit", type=int, default=0, help="限制测试数量，0=全部")
    parser.add_argument("--output", type=str, default="eval_e2e_report.json")
    args = parser.parse_args()

    config = AppConfig()
    llm, retrieval_service, embeddings = build_components(config)

    print(f"\n{'='*60}")
    print(f"统一法务图全链路评估（retrieve → run_skills → generate → guard）")
    print(f"{'='*60}")

    graph = build_graph(llm, retrieval_service, embeddings)

    dataset = ALL_EVAL_DATASET[:args.limit] if args.limit > 0 else ALL_EVAL_DATASET
    results = []

    for i, case in enumerate(dataset):
        print(f"\n  [{i+1}/{len(dataset)}] {case['query'][:60]}...")
        result = asyncio.run(run_case(graph, case))
        results.append({**result, "query": case["query"]})

        completed = result.get("completion", {}).get("completed", False)
        status_icon = "✅" if completed else "❌"
        tool_cov = result.get("tool_chain", {}).get("coverage", 0)
        print(f"    {status_icon} completion={completed}  "
              f"relevance={result['answer_relevance']:.2f}  "
              f"tool_cov={tool_cov:.2f}  "
              f"latency={result['latency']:.1f}s")

    # 汇总
    completed = sum(1 for r in results if r.get("completion", {}).get("completed"))
    avg_relevance = sum(r["answer_relevance"] for r in results) / len(results)
    avg_tool_cov = sum(r.get("tool_chain", {}).get("coverage", 0) for r in results) / len(results)
    avg_latency = sum(r["latency"] for r in results) / len(results)
    errors = [r for r in results if "error" in r]

    summary = {
        "total": len(results),
        "completion_rate": round(completed / len(results), 4),
        "avg_answer_relevance": round(avg_relevance, 4),
        "avg_tool_chain_coverage": round(avg_tool_cov, 4),
        "avg_latency_s": round(avg_latency, 2),
        "errors": len(errors),
    }

    print(f"\n  --- 统一法务图汇总 ---")
    print(f"  Completion Rate:      {summary['completion_rate']:.4f}")
    print(f"  Answer Relevance:     {summary['avg_answer_relevance']:.4f}")
    print(f"  Tool Chain Coverage:  {summary['avg_tool_chain_coverage']:.4f}")
    print(f"  Avg Latency:          {summary['avg_latency_s']:.2f}s")
    print(f"  Errors:               {summary['errors']}")

    report = {"legal": {"summary": summary, "details": results}}

    # 保存报告
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n📄 报告已保存: {args.output}")
    print(f"✅ 全链路评估完成！")


if __name__ == "__main__":
    main()
