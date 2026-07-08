# -*- coding: utf-8 -*-
"""
相似案例检索技能（Similar Case Retrieval）

技术范式：语义向量检索 + MMR 多样性重排 + 案情要素加权融合

为什么不靠 LLM：
    相似度计算、多样性去冗余是确定性的数学过程，用向量空间运算可解释、
    可单测、可复现。LLM 只在最后一步（可选）做归纳总结，不参与检索打分。

检索流程：
    1. 案情向量化（bge embedding，与主 RAG 共用同一 embedding，避免重复加载）
    2. 余弦相似度召回 Top-N 候选
    3. 案情要素重合度加权（场景/工作年限/争议类型等结构化字段）
    4. MMR（Maximal Marginal Relevance）重排，在相关性与多样性间权衡，
       避免返回多条几乎雷同的案例

数据来源：skills/data/labor_cases.json（拟真演示案例，法条对齐《劳动法》107 条）
"""
import os
import json
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from langchain.tools import tool


# ==================== 案例库加载 ====================

_CASE_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "labor_cases.json")


def _load_cases() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """加载案例库，返回 (cases, meta)"""
    try:
        with open(_CASE_DB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("cases", []), data.get("_meta", {})
    except Exception as e:
        print(f"⚠️ [CaseRetriever] 案例库加载失败: {e}")
        return [], {}


# ==================== 向量数学（确定性核心，可单测） ====================

def _cosine_sim_matrix(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """
    计算 query 向量与文档矩阵中每一行的余弦相似度。

    Args:
        query_vec: shape (d,)
        doc_matrix: shape (n, d)

    Returns:
        shape (n,) 的相似度数组，范围约 [-1, 1]
    """
    if doc_matrix.size == 0:
        return np.array([])
    q_norm = np.linalg.norm(query_vec) + 1e-8
    d_norms = np.linalg.norm(doc_matrix, axis=1) + 1e-8
    return (doc_matrix @ query_vec) / (d_norms * q_norm)


def _pairwise_cosine(doc_matrix: np.ndarray) -> np.ndarray:
    """文档两两余弦相似度矩阵，shape (n, n)，用于 MMR 多样性惩罚"""
    if doc_matrix.size == 0:
        return np.array([[]])
    norms = np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-8
    normed = doc_matrix / norms
    return normed @ normed.T


def mmr_select(
    relevance: np.ndarray,
    sim_matrix: np.ndarray,
    top_k: int,
    lambda_param: float = 0.7,
) -> List[int]:
    """
    Maximal Marginal Relevance 重排选择。

    每一步选择使 [λ·相关性 − (1−λ)·与已选集合的最大相似度] 最大的候选，
    在"与查询相关"和"与已选结果不冗余"之间权衡。

    Args:
        relevance: shape (n,) 每个候选与 query 的相关性分数
        sim_matrix: shape (n, n) 候选两两相似度
        top_k: 选取数量
        lambda_param: 权衡系数，1.0=只看相关性，0.0=只看多样性

    Returns:
        选中候选的索引列表（按选择顺序）
    """
    n = len(relevance)
    if n == 0:
        return []
    top_k = min(top_k, n)

    selected: List[int] = []
    candidates = list(range(n))

    # 第一个：直接选相关性最高的
    first = int(np.argmax(relevance))
    selected.append(first)
    candidates.remove(first)

    while len(selected) < top_k and candidates:
        best_idx = -1
        best_score = -np.inf
        for c in candidates:
            # 与已选集合的最大相似度（冗余度）
            max_sim_to_selected = max(sim_matrix[c][s] for s in selected)
            mmr_score = lambda_param * relevance[c] - (1 - lambda_param) * max_sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = c
        selected.append(best_idx)
        candidates.remove(best_idx)

    return selected


# ==================== 案情要素加权（结构化先验） ====================

# 争议类型关键词 → category 的映射，用于从 query 推断意图类别
_CATEGORY_HINTS = {
    "试用期": ["试用期", "试用", "转正"],
    "加班": ["加班", "延长工作时间", "加班费", "超时"],
    "解除": ["解除", "辞退", "开除", "裁员", "违法解除", "单方"],
    "工资": ["工资", "克扣", "拖欠", "欠薪", "最低工资"],
    "社保": ["社保", "社会保险", "五险", "公积金"],
    "竞业": ["竞业", "竞业限制", "竞业禁止"],
    "工伤": ["工伤", "职业病", "因工"],
    "合同形式": ["书面合同", "未签", "无固定期限", "签合同"],
}


def _infer_categories(query: str) -> set:
    """从查询文本推断可能涉及的争议类别（确定性关键词匹配）"""
    matched = set()
    for cat, kws in _CATEGORY_HINTS.items():
        if any(kw in query for kw in kws):
            matched.add(cat)
    return matched


def _element_boost(query: str, case: Dict[str, Any]) -> float:
    """
    案情要素重合度加成 [0, 1]。

    类别命中 + 关键场景词命中，给语义相似度一个结构化先验加权，
    让"同类争议"的案例排名更靠前。
    """
    score = 0.0
    query_cats = _infer_categories(query)
    if case.get("category") in query_cats:
        score += 0.6
    # 场景词命中
    scenario = (case.get("elements") or {}).get("scenario", "")
    if scenario and any(ch in query for ch in scenario):
        score += 0.4
    return min(score, 1.0)


# ==================== 检索器（持有 embedding，惰性向量化案例库） ====================

class CaseRetriever:
    """
    相似案例检索器

    持有一个 embedding 实例（与主 RAG 共用 bge-m3，避免重复加载），
    首次检索时把案例库向量化并缓存。
    """

    def __init__(self, embeddings, alpha: float = 0.75, mmr_lambda: float = 0.7):
        """
        Args:
            embeddings: LangChain embedding 实例（需有 embed_query / embed_documents）
            alpha: 语义相似度与要素加成的融合权重
                   final = alpha * cosine + (1 - alpha) * element_boost
            mmr_lambda: MMR 相关性/多样性权衡系数
        """
        self.embeddings = embeddings
        self.alpha = alpha
        self.mmr_lambda = mmr_lambda
        self.cases, self.meta = _load_cases()
        self._doc_matrix: Optional[np.ndarray] = None

    def _ensure_vectorized(self):
        """惰性向量化案例库（首次检索时执行一次）"""
        if self._doc_matrix is not None or not self.cases:
            return
        # 用"标题 + 案情 + 争议焦点"作为案例的检索表示
        texts = [
            f"{c.get('title','')}。{c.get('facts','')}。{c.get('dispute_focus','')}"
            for c in self.cases
        ]
        vectors = self.embeddings.embed_documents(texts)
        self._doc_matrix = np.array(vectors, dtype=np.float32)

    def retrieve(
        self, query: str, top_k: int = 3, candidate_n: int = 8
    ) -> List[Dict[str, Any]]:
        """
        检索相似案例。

        Args:
            query: 查询文本（用户的纠纷描述）
            top_k: 最终返回数量
            candidate_n: MMR 之前的召回候选数

        Returns:
            案例列表，每条附 relevance / final_score
        """
        if not self.cases:
            return []
        self._ensure_vectorized()

        query_vec = np.array(self.embeddings.embed_query(query), dtype=np.float32)

        # 1. 语义相似度
        cos = _cosine_sim_matrix(query_vec, self._doc_matrix)
        # 归一化到 [0,1]，便于与要素加成融合
        cos_norm = (cos - cos.min()) / (cos.max() - cos.min() + 1e-8)

        # 2. 要素加成融合
        final_scores = np.zeros(len(self.cases), dtype=np.float32)
        for i, case in enumerate(self.cases):
            boost = _element_boost(query, case)
            final_scores[i] = self.alpha * cos_norm[i] + (1 - self.alpha) * boost

        # 3. 取 Top-N 候选
        candidate_n = min(candidate_n, len(self.cases))
        cand_idx = list(np.argsort(final_scores)[::-1][:candidate_n])

        # 4. MMR 在候选内做多样性重排
        cand_matrix = self._doc_matrix[cand_idx]
        cand_relevance = final_scores[cand_idx]
        cand_sim = _pairwise_cosine(cand_matrix)
        mmr_local = mmr_select(cand_relevance, cand_sim, top_k, self.mmr_lambda)
        final_idx = [cand_idx[i] for i in mmr_local]

        results = []
        for i in final_idx:
            case = dict(self.cases[i])
            case["_relevance"] = round(float(cos_norm[i]), 4)
            case["_final_score"] = round(float(final_scores[i]), 4)
            results.append(case)
        return results

    def retrieve_as_string(self, query: str, top_k: int = 3) -> str:
        """检索并格式化为可读文本（供 skill / synthesize 使用）"""
        cases = self.retrieve(query, top_k=top_k)
        if not cases:
            return "未检索到相似案例。"

        disclaimer = self.meta.get("disclaimer", "")
        parts = []
        if disclaimer:
            parts.append(f"【说明】{disclaimer}\n")
        for idx, c in enumerate(cases, 1):
            articles = "、".join(f"第{a}条" for a in c.get("articles", []))
            parts.append(
                f"【相似案例 {idx}】{c.get('title','')}（相关度 {c.get('_final_score')}）\n"
                f"- 争议类型：{c.get('category','')}\n"
                f"- 案情：{c.get('facts','')}\n"
                f"- 争议焦点：{c.get('dispute_focus','')}\n"
                f"- 裁判要点：{c.get('key_points','')}\n"
                f"- 涉及《劳动法》：{articles}\n"
                f"- 结果：{c.get('outcome','')}"
            )
        return "\n\n".join(parts)


# ==================== @tool / 统一接口（与现有 skill 形态一致） ====================

_retriever: Optional[CaseRetriever] = None


def init_skill(embeddings):
    """初始化案例检索器（由 WorkflowService 注入共用 embedding）"""
    global _retriever
    _retriever = CaseRetriever(embeddings)


@tool
def case_retriever(query: str) -> str:
    """相似案例检索工具。输入劳动纠纷描述，返回最相似的历史案例及裁判要点。

    用于：当事人想知道"我这种情况，以前类似的案子是怎么判的"。

    Args:
        query: 劳动纠纷情况描述

    Returns:
        相似案例列表（含案情、裁判要点、涉及法条、结果）
    """
    if not query or not query.strip():
        return json.dumps({"error": "查询为空"}, ensure_ascii=False)
    if _retriever is None:
        return json.dumps(
            {"error": "skill 未初始化，请先调用 init_skill(embeddings)"},
            ensure_ascii=False,
        )
    return _retriever.retrieve_as_string(query)


def case_retriever_skill(query: str, law_context: str = "") -> str:
    """统一接口：接受 query（纠纷描述）"""
    return case_retriever.invoke({"query": query})


# 导出
CASE_RETRIEVER_TOOLS = [case_retriever]
