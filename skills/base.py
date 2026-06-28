# -*- coding: utf-8 -*-
"""
Skill 基础设施

定义统一的 Skill 接口和工具包装函数。
统一接口：fn(query: str, law_context: str = "") -> str
"""
import json
from typing import Callable
from langchain.tools import tool


def create_skill_fn(
    impl_fn: Callable,
    name: str,
) -> Callable:
    """
    包装 skill 实现函数为统一接口

    统一接口：fn(query: str, law_context: str = "") -> str
    内部自动捕获异常，返回 JSON 格式的错误信息。

    Args:
        impl_fn: 实际的 skill 实现函数
        name: skill 名称（用于错误信息）

    Returns:
        统一接口函数
    """
    def skill_fn(query: str, law_context: str = "") -> str:
        try:
            return impl_fn(query, law_context=law_context)
        except TypeError:
            # 兼容旧版单参数 skill
            try:
                return impl_fn(query)
            except Exception as e:
                return json.dumps(
                    {"error": str(e), "skill": name},
                    ensure_ascii=False,
                )
        except Exception as e:
            return json.dumps(
                {"error": str(e), "skill": name},
                ensure_ascii=False,
            )
    return skill_fn


def create_skill_tool(skill_fn: Callable, name: str, description: str):
    """
    将 skill 函数包装为 LangChain Tool（供 ReAct Agent 使用）

    创建的 tool 支持两个参数：
    - query: 用户问题或输入文本
    - law_context: 法律依据（可选，由检索工具提供）

    Args:
        skill_fn: skill 函数 fn(query, law_context="") -> str
        name: tool 名称
        description: tool 描述（ReAct Agent 可见）

    Returns:
        LangChain Tool 对象
    """
    @tool(name=name, description=description)
    def skill_tool(query: str, law_context: str = "") -> str:
        """执行 skill 分析。query 为用户问题，law_context 为可选的法律依据。"""
        try:
            return skill_fn(query, law_context=law_context)
        except Exception as e:
            return json.dumps({"error": str(e), "skill": name}, ensure_ascii=False)
    return skill_tool
