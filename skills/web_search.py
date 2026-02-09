# -*- coding: utf-8 -*-
"""
联网搜索技能

封装 DuckDuckGo 搜索为 LangChain Tool。
"""
from langchain.tools import tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper


@tool
def web_search(query: str) -> str:
    """
    联网搜索工具。用于查询最新的网络信息，如新闻、法规更新、招聘信息等。
    
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


@tool
def job_search(query: str) -> str:
    """
    招聘信息搜索工具。专门用于查找职位、薪资、公司招聘等信息。
    
    Args:
        query: 职位名称或公司名称
    
    Returns:
        招聘相关的搜索结果
    """
    try:
        wrapper = DuckDuckGoSearchAPIWrapper(timeout=15)
        search_query = f"{query} 招聘 最新"
        results = wrapper.run(search_query)
        if not results:
            return "未找到相关招聘信息。"
        return results
    except Exception as e:
        return f"招聘搜索失败: {str(e)}"


# 导出的工具列表
WEB_SEARCH_TOOLS = [web_search, job_search]
