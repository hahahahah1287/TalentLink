# -*- coding: utf-8 -*-
"""
法律文档结构化解析器

按条款切分，保留层级结构信息（法律名称/章节/条款号）。
替代固定字数切分，确保每个 Document 是一个完整的法律条款。
"""
import re
from typing import List, Optional
from langchain_core.documents import Document


class LegalDocumentParser:
    """
    法律文档解析器

    输入：完整的法律文本（如 labor_law.txt）
    输出：按条款切分的 Document 列表，每个带 metadata
    """

    # 法律名称匹配（《书名号》格式）
    LAW_NAME_PATTERN = r'《([^》]+)》'

    # 章节匹配：第X章 + 后续标题
    # 适配全角空格（\u3000）和普通空格混用的格式
    CHAPTER_PATTERN = r'第[一二三四五六七八九十百千]+章[\u3000\s]+\S+'

    # 条款匹配：以「第X条」开头（前面可能有全角空白缩进）
    # 使用正向肯定查找，分割时保留分隔符
    ARTICLE_PATTERN = r'(?=[\s\u3000]*第[一二三四五六七八九十百千零\d]+条[\s\u3000])'

    def __init__(self):
        self.current_law: Optional[str] = None
        self.current_chapter: Optional[str] = None

    def parse(self, text: str, source: str = "unknown") -> List[Document]:
        """
        解析法律文本

        Args:
            text: 完整的法律文本内容
            source: 来源文件名

        Returns:
            List[Document]: 按条款切分的文档列表
        """
        documents: List[Document] = []

        # 1. 提取法律名称
        self.current_law = self._extract_law_name(text, source)

        # 2. 按章节分割
        chapters = self._split_by_chapters(text)

        for chapter_text in chapters:
            # 提取章节标题
            self.current_chapter = self._extract_chapter_title(chapter_text)

            # 在章节内按条款切分
            articles = self._split_by_articles(chapter_text)

            for article_content in articles:
                # 清理前后空白（包括全角空格）
                article_content = article_content.strip().strip('\u3000')
                if not article_content:
                    continue

                # 提取条款号
                article_number = self._extract_article_number(article_content)

                # 跳过非条款内容（如前言、发布日期信息）
                if article_number == "未知条款":
                    continue

                # 构建 Document
                doc = Document(
                    page_content=article_content,
                    metadata={
                        "law": self.current_law,
                        "chapter": self.current_chapter,
                        "article": article_number,
                        "source": source,
                    }
                )
                documents.append(doc)

        return documents

    def _extract_law_name(self, text: str, fallback: str) -> str:
        """提取法律名称"""
        # 优先匹配《书名号》格式
        match = re.search(self.LAW_NAME_PATTERN, text[:500])
        if match:
            return match.group(1)

        # 尝试从文件名提取（去掉路径和扩展名）
        import os
        basename = os.path.splitext(os.path.basename(fallback))[0]
        if basename and basename != "unknown":
            return basename

        return fallback

    def _split_by_chapters(self, text: str) -> List[str]:
        """按章节分割文本"""
        # 如果找不到章节标记，整个文本视为一个章节
        if not re.search(self.CHAPTER_PATTERN, text):
            return [text]

        # 按章节标题分割，保留标题本身
        parts = re.split(f'({self.CHAPTER_PATTERN})', text)

        chapters: List[str] = []
        # parts[0] 是第一章之前的内容（前言/发布信息），跳过
        for i in range(1, len(parts), 2):
            chapter_title = parts[i]
            chapter_body = parts[i + 1] if i + 1 < len(parts) else ""
            chapters.append(chapter_title + chapter_body)

        return chapters if chapters else [text]

    def _extract_chapter_title(self, text: str) -> str:
        """提取章节标题"""
        match = re.search(self.CHAPTER_PATTERN, text)
        if match:
            # 规范化空白：全角空格 → 普通空格，多空格 → 单空格
            title = match.group(0).strip()
            title = re.sub(r'[\u3000\s]+', ' ', title)
            return title
        return "未知章节"

    def _split_by_articles(self, text: str) -> List[str]:
        """按条款分割文本"""
        parts = re.split(self.ARTICLE_PATTERN, text)
        # 过滤掉空字符串和纯章节标题
        return [
            p for p in parts
            if p.strip() and not re.match(self.CHAPTER_PATTERN, p.strip())
        ]

    def _extract_article_number(self, text: str) -> str:
        """提取条款号（第X条）"""
        # 匹配开头的条款号，允许前导空白
        match = re.match(
            r'[\s\u3000]*(第[一二三四五六七八九十百千零\d]+条)',
            text
        )
        return match.group(1) if match else "未知条款"


def parse_legal_document(text: str, source: str = "unknown") -> List[Document]:
    """便捷函数：解析法律文档"""
    parser = LegalDocumentParser()
    return parser.parse(text, source)
