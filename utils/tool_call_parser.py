# -*- coding: utf-8 -*-
"""
Tool Call Parser — 让 ChatLlamaCpp 正确输出 tool_calls

问题：
  ChatLlamaCpp + Qwen 3.5 等模型，bind_tools 虽然生效（模型学会了输出 tool call），
  但模型将 tool call 以纯文本 XML 输出在 content 字段，而非 tool_calls 属性。

  create_react_agent 的 Agent Loop 依赖 tool_calls 做调度：
    if msg.tool_calls → 执行工具
    else → content 就是最终回答

  结果：tool_calls 为空，Agent 把 raw XML 当最终回答返回。

方案：
  包装 ChatLlamaCpp，在 invoke/ainvoke 后自动：
  1. 检查 tool_calls 是否为空
  2. 若为空，从 content 解析 XML 填充到 tool_calls
  3. 清理 content 中的 XML

  create_react_agent 无需改动，自动受益。
"""
import re
from typing import Any, List

from langchain_core.messages import AIMessage
from langchain_core.language_models.base import LanguageModelInput


# ==================== XML 解析 ====================

def _build_patterns():
    """预编译正则（用 chr() 避免编辑器对 XML 标签的误处理）"""
    o, c, s = chr(60), chr(62), chr(47)  # <, >, /

    # 匹配 otool_callc ... o/tool_callc
    tc_re = re.compile(o + r'tool_call' + c + r'(.*?)' + o + r'/tool_call' + c, re.DOTALL)

    # 匹配 ofunction=namec ... o/functionc
    fn_re = re.compile(o + r'function=(\w+)' + c + r'(.*?)' + o + r'/function' + c, re.DOTALL)

    # 匹配 oparameter=keyc ... o/parameterc
    param_re = re.compile(o + r'parameter=(\w+)' + c + r'(.*?)' + o + r'/parameter' + c, re.DOTALL)

    # 匹配 tool call 标签（用于清理 content）
    # 策略：标签名含 tool/call/function，或标签内含 JSON
    tag_re = re.compile(
        o + r'(\w*(?:tool|call|function)\w*)[^' + c + r']*' + c + r'[\s\S]*?' + o + r'/\1' + c
        + r'|'
        + o + r'(\w*(?:tool|call|function)\w*)[^' + c + r']*' + c + r'[\s\S]*',
        re.IGNORECASE
    )

    return tc_re, fn_re, param_re, tag_re


_TC_RE, _FUNC_RE, _PARAM_RE, _TAG_RE = _build_patterns()


def parse_tool_calls_from_content(content: str) -> List[dict]:
    """
    从 LLM 输出的纯文本 XML 中解析 tool call。

    Args:
        content: LLM 输出文本（可能包含 tool call XML）

    Returns:
        LangChain 标准 tool_calls 列表:
        [{"name": str, "args": dict, "id": str, "type": "tool_call"}]
    """
    if not content:
        return []

    tool_calls = []
    for tc_match in _TC_RE.finditer(content):
        tc_block = tc_match.group(1)
        for func_match in _FUNC_RE.finditer(tc_block):
            func_name = func_match.group(1)
            func_body = func_match.group(2)

            args = {}
            for param_match in _PARAM_RE.finditer(func_body):
                key = param_match.group(1)
                value = param_match.group(2).strip()
                args[key] = value

            tool_calls.append({
                "name": func_name,
                "args": args,
                "id": f"call_{func_name}_{len(tool_calls)}",
                "type": "tool_call",
            })

    return tool_calls


def _strip_tool_call_xml(text: str) -> str:
    """移除文本中的 tool call XML 标签，清理多余空行"""
    if not text:
        return ""
    cleaned = _TAG_RE.sub('', text)
    cleaned = re.sub(r'\n{2,}', '\n', cleaned).strip()
    return cleaned


# ==================== LLM 包装层 ====================

class ToolCallParserLLM:
    """
    ChatLlamaCpp 包装层：自动解析 content 中的 tool call XML。

    包装任意 LangChain ChatModel，在 invoke/ainvoke 后：
    1. 如果 AIMessage.tool_calls 为空，尝试从 content 解析
    2. 解析成功则填充 tool_calls 并清理 content
    3. 解析失败则保持原样（正常的文本回复）

    只影响 create_react_agent 的 Agent Loop 调度。
    Plan-and-Execute 链路不经过此层，不受影响。

    Usage:
        raw_llm = ChatLlamaCpp(...)
        llm = ToolCallParserLLM(raw_llm)
        agent = create_react_agent(llm, tools=...)  # 自动受益
    """

    def __init__(self, llm):
        self._llm = llm

    def _parse_response(self, msg: AIMessage) -> AIMessage:
        """解析单条 AIMessage 的 tool call"""
        # 已有 tool_calls，无需解析
        if msg.tool_calls:
            return msg

        # 尝试从 content 解析 tool call XML
        content = msg.content or ""
        parsed_calls = parse_tool_calls_from_content(content)

        if parsed_calls:
            # 解析成功：填充 tool_calls，清理 content
            clean_content = _strip_tool_call_xml(content)
            return AIMessage(
                content=clean_content,
                tool_calls=parsed_calls,
                additional_kwargs=msg.additional_kwargs,
                response_metadata=msg.response_metadata,
            )

        return msg

    def invoke(self, input: LanguageModelInput, **kwargs: Any) -> AIMessage:
        result = self._llm.invoke(input, **kwargs)
        if isinstance(result, AIMessage):
            return self._parse_response(result)
        return result

    async def ainvoke(self, input: LanguageModelInput, **kwargs: Any) -> AIMessage:
        result = await self._llm.ainvoke(input, **kwargs)
        if isinstance(result, AIMessage):
            return self._parse_response(result)
        return result

    # 透传 bind_tools：包装后的 bound LLM 也需要解析能力
    def bind_tools(self, tools, **kwargs):
        bound = self._llm.bind_tools(tools, **kwargs)
        return ToolCallParserLLM(bound)

    # 透传 with_structured_output
    def with_structured_output(self, schema, **kwargs):
        return self._llm.with_structured_output(schema, **kwargs)

    # 透传其他属性（model_name、temperature 等）
    def __getattr__(self, name):
        return getattr(self._llm, name)
