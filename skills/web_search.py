# -*- coding: utf-8 -*-
"""
联网搜索技能

封装 DuckDuckGo 搜索为 LangChain Tool。

纯法务定位下，联网检索只作为“最新政策/地方标准”的 freshness 线索：
- 优先检索官方域名；
- 输出结构化 URL evidence；
- 明确标记 trusted / untrusted；
- 永远不把外部网页摘要冒充本地法条证据。
"""
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from langchain.tools import tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from utils.evidence import stable_text_hash
from utils.skill_result import make_skill_result


SOURCE_WHITELIST_VERSION = "official-cn-web-v1"

# 法务 freshness 场景只信官方/准官方入口。地方人社局、人民政府站点通常都在 gov.cn 下。
_TRUSTED_DOMAIN_SUFFIXES = (
    "gov.cn",
)

_PUBLISHER_BY_DOMAIN = {
    "mohrss.gov.cn": "人力资源和社会保障部",
    "12333.gov.cn": "全国人社政务服务平台",
    "npc.gov.cn": "全国人大",
    "court.gov.cn": "人民法院",
    "gov.cn": "政府/主管部门网站",
}


def _hostname(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower().strip(".")
    except Exception:
        return ""


def _domain_matches(host: str, suffix: str) -> bool:
    host = (host or "").lower().strip(".")
    suffix = (suffix or "").lower().strip(".")
    return bool(host and suffix and (host == suffix or host.endswith("." + suffix)))


def _is_trusted_official_domain(url: str) -> bool:
    """Return True only for exact/subdomain matches, avoiding evilgov.cn false positives."""
    host = _hostname(url)
    return any(_domain_matches(host, suffix) for suffix in _TRUSTED_DOMAIN_SUFFIXES)


def _publisher_for_url(url: str) -> str:
    host = _hostname(url)
    for suffix, publisher in _PUBLISHER_BY_DOMAIN.items():
        if _domain_matches(host, suffix):
            return publisher
    return host or "未知来源"


def _official_query(query: str) -> str:
    cleaned = (query or "").strip()
    if not cleaned:
        return cleaned
    if "site:" in cleaned.lower():
        return cleaned
    return f"{cleaned} site:gov.cn"


def _raw_search_results(query: str, max_results: int = 5) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Return structured DDG results where supported; fall back to plain summary."""
    try:
        wrapper = DuckDuckGoSearchAPIWrapper(timeout=15)
        if hasattr(wrapper, "results"):
            try:
                return wrapper.results(query, max_results=max_results), None
            except TypeError:
                return wrapper.results(query, max_results), None
        summary = wrapper.run(query)
        if not summary:
            return [], None
        return [{"title": "DuckDuckGo 摘要", "snippet": summary, "link": ""}], None
    except Exception as e:
        return [], f"搜索失败: {str(e)}"


def _normalize_result(raw: Dict[str, Any], query: str, retrieved_at: str) -> Dict[str, Any]:
    url = raw.get("link") or raw.get("href") or raw.get("url") or ""
    title = raw.get("title") or raw.get("name") or "未命名结果"
    snippet = raw.get("snippet") or raw.get("body") or raw.get("description") or ""
    trusted = _is_trusted_official_domain(url)
    publisher = _publisher_for_url(url)
    content_hash = stable_text_hash("\n".join([title, url, snippet]))
    return {
        "source_kind": "external_url" if url else "external_search_summary",
        "url": url,
        "title": title,
        "snippet": snippet,
        "domain": _hostname(url),
        "publisher": publisher,
        "trusted": trusted,
        "retrieved_at": retrieved_at,
        "query": query,
        "content_hash": content_hash,
        "whitelist_version": SOURCE_WHITELIST_VERSION,
        "note": (
            "官方白名单来源，仅作为最新政策/地方标准线索；引用前仍需核对正文、生效时间和适用地区。"
            if trusted else
            "非官方白名单来源，不纳入可信法律依据。"
        ),
    }


def _search_official_sources(query: str, max_results: int = 5) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Search official sources first and return only whitelist-trusted URL evidence."""
    retrieved_at = datetime.now(timezone.utc).isoformat()
    search_query = _official_query(query)
    raw_results, error = _raw_search_results(search_query, max_results=max_results)
    normalized = [_normalize_result(r, query, retrieved_at) for r in raw_results]
    trusted = [r for r in normalized if r.get("trusted")]

    meta = {
        "provider": "duckduckgo",
        "search_query": search_query,
        "retrieved_at": retrieved_at,
        "official_only": True,
        "whitelist_version": SOURCE_WHITELIST_VERSION,
        "raw_result_count": len(raw_results),
        "trusted_result_count": len(trusted),
    }
    if error:
        meta["error"] = error
    return trusted[:max_results], meta


def _format_result_lines(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "未命中官方白名单来源。请人工到人社部、地方人社局或政府官网核验。"

    lines = []
    for idx, item in enumerate(results, 1):
        snippet = (item.get("snippet") or "").strip()
        if len(snippet) > 220:
            snippet = snippet[:220] + "..."
        lines.append(
            f"{idx}. 【官方线索】{item.get('title', '未命名结果')}\n"
            f"   来源：{item.get('publisher', '未知来源')} | {item.get('url', '')}\n"
            f"   摘要：{snippet or '（无摘要）'}"
        )
    return "\n".join(lines)


@tool
def web_search(query: str) -> str:
    """
    联网搜索工具。用于查询最新政策、地方标准等 freshness 信息。

    Args:
        query: 搜索关键词

    Returns:
        官方白名单 URL 线索摘要；不会返回非官方来源作为法律依据。
    """
    results, meta = _search_official_sources(query)
    if meta.get("error"):
        return meta["error"]
    return _format_result_lines(results)


# ==================== 统一接口 ====================

def web_search_skill(query: str, law_context: str = "") -> str:
    """统一接口：联网搜索。输出统一 SkillResult，不把网页摘要冒充本地法源。"""
    results, meta = _search_official_sources(query)
    raw_display = _format_result_lines(results)
    is_error = bool(meta.get("error"))
    status = "error" if is_error else ("ok" if results else "no_official_source")

    findings = [
        {
            "finding_type": "freshness_official_url",
            "title": item.get("title"),
            "url": item.get("url"),
            "publisher": item.get("publisher"),
            "snippet": item.get("snippet"),
            "trusted": item.get("trusted"),
            "retrieved_at": item.get("retrieved_at"),
        }
        for item in results
    ]

    display_prefix = (
        "【外部检索提示】以下为官方白名单 URL 线索，仅用于最新政策/地方标准核验；"
        "不能替代正式法律文本或本地 evidence，回答时应说明需以官网原文为准。\n"
    )
    if is_error:
        display_prefix += f"检索失败：{meta.get('error')}\n"

    unified = make_skill_result(
        skill_name="web_search",
        findings=findings,
        evidence=results,
        provenance=meta,
        display_text=display_prefix + raw_display,
        metrics={
            "external": True,
            "official_only": True,
            "result_count": len(results),
            "trusted_result_count": len(results),
            "raw_result_count": meta.get("raw_result_count", 0),
            "whitelist_version": SOURCE_WHITELIST_VERSION,
        },
        legacy={
            "raw_search_result": raw_display,
            "official_results": results,
            "source_whitelist_version": SOURCE_WHITELIST_VERSION,
        },
        status=status,
    )
    return json.dumps(unified, ensure_ascii=False, indent=2)


# 导出的工具列表
WEB_SEARCH_TOOLS = [web_search]
