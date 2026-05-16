# -*- coding: utf-8 -*-
"""
求职搜索工具

封装 DuckDuckGo 搜索为可调用函数。
"""
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper


def web_search(query: str, timeout: int = 15) -> str:
    """
    联网搜索

    Args:
        query: 搜索关键词
        timeout: 超时时间

    Returns:
        搜索结果摘要
    """
    try:
        wrapper = DuckDuckGoSearchAPIWrapper(timeout=timeout)
        results = wrapper.run(query)
        if not results:
            return "未找到相关搜索结果。"
        return results
    except Exception as e:
        return f"搜索失败: {str(e)}"


def job_search(query: str, timeout: int = 15) -> str:
    """
    招聘信息搜索

    Args:
        query: 职位名称或公司名称
        timeout: 超时时间

    Returns:
        招聘相关的搜索结果
    """
    try:
        wrapper = DuckDuckGoSearchAPIWrapper(timeout=timeout)
        search_query = f"{query} 招聘 最新"
        results = wrapper.run(search_query)
        if not results:
            return "未找到相关招聘信息。"
        return results
    except Exception as e:
        return f"招聘搜索失败: {str(e)}"
