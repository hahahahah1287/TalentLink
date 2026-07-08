# -*- coding: utf-8 -*-
"""
Evidence schema and helpers for citation-grounded legal RAG.

The runtime still passes LangChain Document objects around, but every retrieved
article can now be normalized into a stable, auditable evidence dict.  The
helpers are intentionally dependency-light so they work with both freshly parsed
Documents and old FAISS indexes whose metadata is sparse.
"""
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document


@dataclass
class LegalSourceRef:
    """Normalized legal source identity for one article."""

    source_type: str
    law_title: str
    canonical_law_title: str
    article_number: Optional[int]
    article_label: str
    canonical_citation: str
    source_path: str = ""


@dataclass
class EvidenceItem:
    """Article-level evidence carried through retrieval, generation and guard."""

    evidence_id: str
    article_id: str
    source: LegalSourceRef
    text: str
    text_hash: str
    retrieval_score: Optional[float] = None
    rank: Optional[int] = None
    source_kind: str = "retrieved_article"


_CN_NUM_MAP = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CN_UNIT_MAP = {"十": 10, "百": 100, "千": 1000, "万": 10000}

_LAW_TITLE_ALIASES = {
    "labor_law": ("劳动法", "中华人民共和国劳动法", "labor_law"),
    "中华人民共和国劳动法": ("劳动法", "中华人民共和国劳动法", "labor_law"),
    "劳动法": ("劳动法", "中华人民共和国劳动法", "labor_law"),
}


def _cn_to_int(text: str) -> Optional[int]:
    """Convert simple Chinese numerals such as 一百零七 to int."""
    if not text:
        return None
    if text.isdigit():
        return int(text)

    total = 0
    section = 0
    number = 0
    seen = False
    for char in text:
        if char in _CN_NUM_MAP:
            number = _CN_NUM_MAP[char]
            seen = True
        elif char in _CN_UNIT_MAP:
            unit = _CN_UNIT_MAP[char]
            seen = True
            if unit == 10000:
                section = (section + (number or 1)) * unit
                total += section
                section = 0
            else:
                section += (number or 1) * unit
            number = 0
        else:
            continue
    if not seen:
        return None
    return total + section + number


def normalize_article_number(value: Any) -> Optional[int]:
    """Extract an article number from labels like 第三十八条 / 第38条."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"第\s*([一二两三四五六七八九十百千万零〇\d]+)\s*条", text)
    if match:
        return _cn_to_int(match.group(1))
    if text.isdigit():
        return int(text)
    return _cn_to_int(text)


def extract_article_label(text: str, fallback_number: Optional[int] = None) -> str:
    """Return the original article label if present, else a stable numeric label."""
    match = re.search(r"第\s*([一二两三四五六七八九十百千万零〇\d]+)\s*条", text or "")
    if match:
        return f"第{match.group(1)}条"
    if fallback_number is not None:
        return f"第{fallback_number}条"
    return "未知条款"


def normalize_law_title(raw_title: str = "", source_path: str = "") -> Dict[str, str]:
    """Normalize law names and provide a stable slug for article IDs."""
    title = (raw_title or "").strip()
    basename = os.path.splitext(os.path.basename(source_path or ""))[0]
    key = title or basename
    if basename == "labor_law" or key in _LAW_TITLE_ALIASES:
        short, canonical, slug = _LAW_TITLE_ALIASES.get(key, _LAW_TITLE_ALIASES["labor_law"])
        return {"law_title": short, "canonical_law_title": canonical, "law_slug": slug}

    canonical = title or basename or "未知法律"
    slug_source = canonical.encode("utf-8")
    slug = hashlib.sha1(slug_source).hexdigest()[:10]
    return {"law_title": canonical, "canonical_law_title": canonical, "law_slug": slug}


def stable_text_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:12]


def build_article_id(canonical_law_title: str, article_number: Optional[int], source_path: str = "") -> str:
    title_info = normalize_law_title(canonical_law_title, source_path)
    suffix = str(article_number) if article_number is not None else "unknown"
    return f"{title_info['law_slug']}:{suffix}"


def build_canonical_citation(canonical_law_title: str, article_label: str) -> str:
    if not canonical_law_title or canonical_law_title == "未知法律":
        return article_label or "未知条款"
    if not article_label or article_label == "未知条款":
        return f"《{canonical_law_title}》"
    return f"《{canonical_law_title}》{article_label}"


def document_to_evidence(
    doc: Document,
    rank: Optional[int] = None,
    retrieval_score: Optional[float] = None,
    evidence_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize a LangChain Document into a serializable EvidenceItem dict."""
    metadata = dict(doc.metadata or {})
    source_path = metadata.get("source_path") or metadata.get("source") or ""
    law_info = normalize_law_title(
        metadata.get("canonical_law_title") or metadata.get("law_title") or metadata.get("law", ""),
        source_path,
    )
    article_number = (
        metadata.get("article_number")
        or normalize_article_number(metadata.get("article"))
        or normalize_article_number(doc.page_content)
    )
    article_label = metadata.get("article_label") or extract_article_label(
        metadata.get("article", "") or doc.page_content,
        article_number,
    )
    canonical_citation = metadata.get("canonical_citation") or build_canonical_citation(
        law_info["canonical_law_title"],
        article_label,
    )
    article_id = metadata.get("article_id") or build_article_id(
        law_info["canonical_law_title"],
        article_number,
        source_path,
    )
    text_hash = metadata.get("text_hash") or stable_text_hash(doc.page_content)

    source = LegalSourceRef(
        source_type=metadata.get("source_type", "statute"),
        law_title=law_info["law_title"],
        canonical_law_title=law_info["canonical_law_title"],
        article_number=article_number,
        article_label=article_label,
        canonical_citation=canonical_citation,
        source_path=source_path,
    )
    resolved_evidence_id = evidence_id or (f"E{rank}" if rank is not None else article_id)
    item = EvidenceItem(
        evidence_id=resolved_evidence_id,
        article_id=article_id,
        source=source,
        text=doc.page_content,
        text_hash=text_hash,
        retrieval_score=retrieval_score,
        rank=rank,
    )
    return asdict(item)


def documents_to_evidence(docs: List[Document]) -> List[Dict[str, Any]]:
    return [document_to_evidence(doc, rank=i + 1, evidence_id=f"E{i + 1}") for i, doc in enumerate(docs)]


def render_evidence_context(evidence_items: List[Dict[str, Any]], separator: str = "\n\n") -> str:
    """Render evidence blocks for prompts and guardrails."""
    if not evidence_items:
        return "未找到相关内容。"
    parts = []
    for item in evidence_items:
        source = item.get("source", {}) or {}
        evidence_id = item.get("evidence_id") or item.get("article_id") or "E?"
        citation = source.get("canonical_citation") or item.get("article_id") or "未知来源"
        text = (item.get("text") or "").strip()
        parts.append(f"【{evidence_id}】{citation}\n{text}")
    return separator.join(parts)


def evidence_article_numbers(evidence_items: List[Dict[str, Any]]) -> set:
    numbers = set()
    for item in evidence_items or []:
        source = item.get("source", {}) or {}
        number = source.get("article_number")
        if isinstance(number, int):
            numbers.add(number)
        else:
            normalized = normalize_article_number(number)
            if normalized is not None:
                numbers.add(normalized)
    return numbers


def evidence_citations(evidence_items: List[Dict[str, Any]]) -> List[str]:
    citations = []
    for item in evidence_items or []:
        source = item.get("source", {}) or {}
        citation = source.get("canonical_citation")
        if citation:
            citations.append(citation)
    return citations


def evidence_law_titles(evidence_items: List[Dict[str, Any]]) -> set:
    titles = set()
    for item in evidence_items or []:
        source = item.get("source", {}) or {}
        for key in ("law_title", "canonical_law_title"):
            title = source.get(key)
            if title:
                titles.add(title)
    return titles


def _walk_json(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def extract_skill_evidence(skill_name: str, raw_output: str) -> List[Dict[str, Any]]:
    """Extract rule/skill legal bases without pretending they were retrieved."""
    try:
        parsed = json.loads(raw_output)
    except Exception:
        return []

    items = []
    seen = set()
    for obj in _walk_json(parsed):
        basis = obj.get("legal_basis") or obj.get("basis")
        articles = obj.get("articles") or []
        rule_id = obj.get("rule_id") or obj.get("id")
        if not basis and not articles:
            continue
        if isinstance(articles, int):
            articles = [articles]
        key = (str(basis), tuple(articles), str(rule_id))
        if key in seen:
            continue
        seen.add(key)
        items.append({
            "source_kind": "skill_basis",
            "skill_name": skill_name,
            "rule_id": rule_id,
            "legal_basis": basis,
            "articles": articles,
            "note": "规则/知识卡片内置依据；未必来自本轮检索证据。",
        })
    return items


def render_skill_evidence(skill_evidence: List[Dict[str, Any]]) -> str:
    if not skill_evidence:
        return ""
    lines = ["【Skill 内置依据（不等同于本轮检索证据）】"]
    for idx, item in enumerate(skill_evidence, 1):
        basis = item.get("legal_basis") or "未标注具体依据"
        rule_id = item.get("rule_id") or "-"
        skill = item.get("skill_name") or "unknown"
        articles = item.get("articles") or []
        article_text = f"；articles={articles}" if articles else ""
        lines.append(f"- S{idx} {skill} rule={rule_id}: {basis}{article_text}")
    return "\n".join(lines)
