# -*- coding: utf-8 -*-
"""
法律术语解释技能（知识图谱实体链接 + Grounding 抗幻觉）

技术范式：轻量知识图谱（术语节点 + 关系边） + 实体链接 + 检索增强 grounding

为什么不靠 LLM 直接解释：
    LLM 凭记忆解释法律术语容易"一本正经地编"——把法条号、补偿倍数说错。
    这里把术语沉淀成结构化知识图谱（skills/data/legal_terms_graph.json）：
      - 实体链接：从用户文本中识别术语（支持别名/模糊匹配）
      - 关系扩展：沿图的边补充"易混淆/常伴随/计算依赖"的关联术语
        （例：问"经济补偿金"自动带出易混的"赔偿金"、伴随的"代通知金"）
      - Grounding：把命中的知识卡片作为唯一事实来源喂给 LLM，
        要求其只能基于卡片解释，不得引入卡片外的法条 → 抗幻觉
    LLM 只做"把结构化卡片组织成通俗语言"，事实由图谱锚定，可追溯到法条。

无 LLM 时：直接返回结构化卡片（仍然可用、准确）。
LLM grounding 是可选增强层。
"""
import os
import json
from typing import Dict, Any, List, Optional, Tuple

from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==================== 知识图谱加载 ====================

_GRAPH_PATH = os.path.join(os.path.dirname(__file__), "data", "legal_terms_graph.json")


def _load_graph() -> Dict[str, Any]:
    try:
        with open(_GRAPH_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("terms", {})
    except Exception as e:
        print(f"⚠️ [TermExplainer] 知识图谱加载失败: {e}")
        return {}


_GRAPH = _load_graph()
# 别名 → 规范术语名 的反向索引（实体链接用）
_ALIAS_INDEX: Dict[str, str] = {}
for _name, _info in _GRAPH.items():
    _ALIAS_INDEX[_name] = _name
    for _alias in _info.get("aliases", []):
        _ALIAS_INDEX[_alias] = _name


# ==================== 实体链接（确定性，可单测） ====================

def link_entities(text: str) -> List[str]:
    """
    从文本中链接到知识图谱中的术语节点。

    策略：精确名/别名匹配优先；按匹配词长度降序，避免短词覆盖长词
    （"竞业" 命中 "竞业限制" 节点）。

    Returns:
        命中的规范术语名列表（去重，保序）
    """
    hits: List[str] = []
    seen = set()
    # 按 key 长度降序，长术语优先
    for surface in sorted(_ALIAS_INDEX.keys(), key=len, reverse=True):
        if surface in text:
            canonical = _ALIAS_INDEX[surface]
            if canonical not in seen:
                seen.add(canonical)
                hits.append(canonical)
    return hits


def expand_related(term: str, max_related: int = 3) -> List[Dict[str, str]]:
    """
    沿知识图谱的边扩展关联术语。

    Returns:
        [{term, rel, note}, ...]
    """
    info = _GRAPH.get(term, {})
    related = []
    for edge in info.get("edges", [])[:max_related]:
        related.append({
            "term": edge["to"],
            "rel": edge.get("rel", ""),
            "note": edge.get("note", ""),
        })
    return related


def build_knowledge_cards(text: str) -> List[Dict[str, Any]]:
    """
    实体链接 + 关系扩展，构建知识卡片集合（grounding 的事实来源）。
    """
    cards = []
    linked = link_entities(text)
    for term in linked:
        info = _GRAPH.get(term, {})
        cards.append({
            "term": term,
            "plain": info.get("plain", ""),
            "articles": info.get("articles", []),
            "law_title": info.get("law_title", "中华人民共和国劳动法"),
            "related": expand_related(term),
        })
    return cards


def _cards_to_context(cards: List[Dict[str, Any]]) -> str:
    """把知识卡片渲染为 grounding 上下文文本"""
    parts = []
    for c in cards:
        law_title = c.get("law_title", "中华人民共和国劳动法")
        articles = "、".join(f"《{law_title}》第{a}条" for a in c.get("articles", [])) or "（本卡片未关联具体法条）"
        rel_lines = "".join(
            f"\n  · 关联「{r['term']}」({r['rel']}): {r['note']}" for r in c.get("related", [])
        )
        parts.append(f"- 术语「{c['term']}」：{c['plain']}\n  法条：{articles}{rel_lines}")
    return "\n".join(parts)


# ==================== LLM Grounding 链（可选增强） ====================

GROUNDING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是法律术语讲解员。**只能依据下面提供的【知识卡片】**用通俗语言解释术语。

严格约束：
- 不得引入知识卡片之外的法条号或数字结论；卡片没有的，不要编。
- 如果用户问到的术语不在卡片中，明说"知识库中暂无该术语"。
- 对易混淆的关联术语（卡片中标注 contrast 的），主动点出区别。
- 语言通俗，可举生活化例子，但法条与倍数必须与卡片一致。"""),
    ("user", """【知识卡片】
{cards}

【用户问题】
{query}

请基于卡片解释：""")
])


def create_grounding_chain(llm):
    return GROUNDING_PROMPT | llm | StrOutputParser()


# ==================== @tool / 统一接口 ====================

_llm = None
_grounding_chain = None


def init_skill(llm):
    """初始化 grounding 链（可选，由 WorkflowService 调用）。不调用则返回结构化卡片。"""
    global _llm, _grounding_chain
    _llm = llm
    _grounding_chain = create_grounding_chain(llm)


@tool
def legal_term_explainer(text: str) -> str:
    """法律术语解释工具。识别文本中的劳动法术语，基于知识图谱解释并扩展关联术语。

    采用"知识图谱实体链接 + grounding"，解释严格锚定法条，避免幻觉。

    Args:
        text: 含法律术语的文本（合同条款、用户问题等）

    Returns:
        术语解释（含通俗释义、法条出处、易混淆/关联术语）
    """
    if not text or not text.strip():
        return json.dumps({"error": "输入文本为空"}, ensure_ascii=False)

    cards = build_knowledge_cards(text)

    if not cards:
        return json.dumps({
            "found_terms": 0,
            "note": "未识别到知识库中的法律术语。",
            "supported_terms": list(_GRAPH.keys()),
        }, ensure_ascii=False, indent=2)

    # 有 LLM → grounding 解释；无 LLM → 返回结构化卡片
    if _grounding_chain is not None:
        try:
            explanation = _grounding_chain.invoke({
                "cards": _cards_to_context(cards),
                "query": text,
            })
            return json.dumps({
                "found_terms": len(cards),
                "linked_terms": [c["term"] for c in cards],
                "explanation": explanation.strip(),
                "grounded_on": [
                    {"term": c["term"], "articles": c["articles"]} for c in cards
                ],
                "note": "解释基于知识图谱卡片生成，法条已锚定。",
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ [TermExplainer] grounding 失败，返回结构化卡片: {e}")

    # fallback：结构化卡片
    return json.dumps({
        "found_terms": len(cards),
        "terms": cards,
        "note": "以下为知识图谱结构化卡片（未经 LLM 改写），法条以《劳动法》为准。",
    }, ensure_ascii=False, indent=2)


def legal_term_skill(query: str, law_context: str = "") -> str:
    """统一接口：接受 query"""
    return legal_term_explainer.invoke({"text": query})


# 导出
LEGAL_TERM_TOOLS = [legal_term_explainer]
