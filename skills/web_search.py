# -*- coding: utf-8 -*-
"""
联网搜索技能

封装 DuckDuckGo 搜索为 LangChain Tool。

纯法务定位下，联网检索不在确定性法务图的主链路里（法条来自本地《劳动法》知识库），
此工具保留作为可选的"查最新政策/新闻"补充能力，不参与意图路由。
"""
from langchain.tools import tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper


@tool
def web_search(query: str) -> str:
    """
    联网搜索工具。用于查询最新的网络信息，如新闻、法规更新等。

    Args:
        query: 搜索关键词

    Returns:
        搜索结果摘要
    """
    try:
        wrapper = DuckDuckGoSearchAPIWrapper(timeout=15)
        results = wrapper.run(query)
        if not results:
            return "未找到相关搜索结果。"
        return results
    except Exception as e:
        return f"搜索失败: {str(e)}"


# ==================== 统一接口 ====================

def web_search_skill(query: str) -> str:
    """统一接口：联网搜索"""
    return web_search.invoke({"query": query})


# 导出的工具列表
WEB_SEARCH_TOOLS = [web_search]
