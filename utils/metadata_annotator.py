# -*- coding: utf-8 -*-
"""
元数据标注器

使用 LLM 为法律条款生成摘要、关键词等结构化元数据。
标注结果会持久化到 FAISS 索引中，下次启动时直接加载。
"""
import json
import time
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate


class LegalMetadataAnnotator:
    """
    法律元数据标注器

    为每个法律条款生成:
    - summary: 一句话摘要
    - keywords: 关键词列表（3-5个）
    - applicable_scenario: 适用场景
    - legal_effect: 法律效力类型
    """

    # 最大重试次数
    MAX_RETRIES = 1

    def __init__(self, llm):
        """
        Args:
            llm: LangChain LLM 实例
        """
        self.llm = llm

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的法律文档分析助手。
请分析以下法律条款，提取关键信息并以 JSON 格式返回。

所属法律：{law_name}
所属章节：{chapter}

需要提取的字段：
- summary: 一句话摘要（50字以内）
- keywords: 关键词列表（3-5个，用于检索匹配）
- applicable_scenario: 适用场景列表（如"调岗"、"加班费"、"解除合同"等）
- legal_effect: 法律效力类型（"强制性"/"指导性"/"程序性"）

返回格式必须是合法的 JSON：
{{
  "summary": "...",
  "keywords": ["...", "..."],
  "applicable_scenario": ["...", "..."],
  "legal_effect": "..."
}}"""),
            ("user", "法律条款内容：\n\n{article_content}\n\n请分析并返回 JSON：")
        ])

        self.chain = self.prompt | self.llm | JsonOutputParser()

    def annotate(self, doc: Document) -> Document:
        """
        为单个文档添加元数据标注

        Args:
            doc: 待标注的 Document

        Returns:
            更新后的 Document（原地修改并返回）
        """
        last_error: Optional[Exception] = None

        for attempt in range(1 + self.MAX_RETRIES):
            try:
                metadata = self.chain.invoke({
                    "article_content": doc.page_content,
                    "law_name": doc.metadata.get("law", "未知法律"),
                    "chapter": doc.metadata.get("chapter", "未知章节"),
                })

                # 合并到原有 metadata
                doc.metadata.update({
                    "summary": metadata.get("summary", ""),
                    "keywords": metadata.get("keywords", []),
                    "applicable_scenario": metadata.get("applicable_scenario", []),
                    "legal_effect": metadata.get("legal_effect", "未知"),
                })
                return doc

            except Exception as e:
                last_error = e
                if attempt < self.MAX_RETRIES:
                    print(f"⚠️ 标注重试 ({attempt + 1}/{self.MAX_RETRIES}): "
                          f"{doc.metadata.get('article', '未知条款')}")
                    time.sleep(1)  # 重试前等待 1s，避免速率限制

        # 所有重试均失败，设置默认值
        print(f"⚠️ 标注失败: {doc.metadata.get('article', '未知条款')} - {last_error}")
        doc.metadata.update({
            "summary": "",
            "keywords": [],
            "applicable_scenario": [],
            "legal_effect": "未知",
        })
        return doc

    def annotate_batch(self, docs: List[Document]) -> List[Document]:
        """
        批量标注

        Args:
            docs: 待标注的 Document 列表

        Returns:
            标注后的 Document 列表
        """
        result: List[Document] = []
        total = len(docs)

        for i, doc in enumerate(docs, 1):
            pct = i * 100 // total
            article = doc.metadata.get("article", "未知条款")
            print(f"📋 标注进度: {i}/{total} ({pct}%) - {article}")
            annotated = self.annotate(doc)
            result.append(annotated)

        return result


def annotate_documents(docs: List[Document], llm) -> List[Document]:
    """便捷函数：批量标注文档"""
    annotator = LegalMetadataAnnotator(llm)
    return annotator.annotate_batch(docs)
