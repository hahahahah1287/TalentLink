# -*- coding: utf-8 -*-
"""
法务合成链

封装确定性法务图最终"合成"节点的 LLM 调用。

LLM 在这里的角色被收窄为**只做合成**：把检索到的法条 + 确定性 skill 已经
判定好的结构化结果，组织成一段通俗、有依据的回答。判定（试用期是否超期、
是否违规、时效是否届满）已由 skill 的规则引擎/状态机完成，不交给 LLM 自由发挥。
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def create_contract_chain(llm):
    """
    创建合同审查 LLM 链

    Args:
        llm: LangChain LLM 实例

    Returns:
        可调用的 chain（接受 dict，返回 str）
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一名专业法务助手。
请基于以下法律法规库、用户上传的合同内容，以及【已识别的风险/合规要点】，专业地回答用户的问题。

要求：
- 【法律法规参考】中的每条证据以【E编号】开头，这是本轮可引用的检索证据。
- 引用法条时，优先写成“根据【E1】《...》第...条”，只引用【法律法规参考】中实际出现的证据，不要编造条文号。
- 【已识别的风险/合规要点】由确定性规则引擎给出，请直接采信并用通俗语言转述，不要推翻或另算结论。
- Skill 中标注的“内置依据”不等于本轮检索证据；如果它不在【法律法规参考】里，表述为“规则依据提示”，不要伪装成已检索法条。
- 如果法律库中没有相关条款，明确说明并给出一般性建议。"""),
        ("user", """
【历史对话】
{history}

【法律法规参考】
{law}

【合同内容】
{contract}

【已识别的风险/合规要点】
{skill_findings}

【用户问题】
{question}

请给出专业分析：""")
    ])
    return prompt | llm | StrOutputParser()


def create_legal_qa_chain(llm):
    """
    创建法律咨询合成链（无合同的纯咨询场景）。

    把检索到的法条 + 确定性 skill 的结构化结论，合成为通俗回答。
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一名专业的劳动法务助手，只回答劳动法相关问题。
请基于以下法律法规和【已识别的要点】，简洁、准确地回答用户问题。

要求：
- 【法律法规参考】中的每条证据以【E编号】开头，这是本轮可引用的检索证据。
- 引用法条时，优先写成“根据【E1】《...》第...条”，只引用【法律法规参考】中实际出现的证据，不要编造。
- 【已识别的要点】来自确定性分析（合规判定、时效计算、术语解释、相似案例等），请直接采信并转述，不要另算或推翻。
- Skill 中标注的“内置依据”不等于本轮检索证据；如果它不在【法律法规参考】里，表述为“规则依据提示”。
- 如果参考资料不足以回答，说明并给出一般性建议；超出劳动法范围的问题，礼貌说明你只处理劳动法务。"""),
        ("user", """
【历史对话】
{history}

【法律法规参考】
{law}

【已识别的要点】
{skill_findings}

【用户问题】
{question}

请回答：""")
    ])
    return prompt | llm | StrOutputParser()


def create_synthesis_chain(llm):
    """
    创建通用结果合成链（用于求职等场景的最终答案生成）

    Args:
        llm: LangChain LLM 实例

    Returns:
        可调用的 chain
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个有用、友好的 AI 助手。
请根据以下信息，简洁、准确地回答用户问题。
如果有检索到的参考资料，请优先基于资料回答；如果资料不足，请说明并给出一般性建议。"""),
        ("user", """
【历史对话】
{history}

【参考资料】
{context}

【用户问题】
{question}

请回答：""")
    ])
    return prompt | llm | StrOutputParser()
