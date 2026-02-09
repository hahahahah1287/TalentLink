# -*- coding: utf-8 -*-
"""
文本分词工具
"""
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter


def create_text_splitter(
    splitter_type: str = "legal",
    chunk_size: int = None,
    chunk_overlap: int = None
) -> RecursiveCharacterTextSplitter:
    """
    工厂函数：根据文档类型创建分词器
    
    使用 LangChain 标准的 RecursiveCharacterTextSplitter，
    避免手写递归逻辑的潜在 bug。
    
    Args:
        splitter_type: 文档类型 - legal(法律), contract(合同), general(通用)
        chunk_size: 自定义 chunk 大小
        chunk_overlap: 自定义重叠大小
    
    Returns:
        配置好的分词器实例
    """
    configs = {
        "legal": {
            "chunk_size": 600,
            "chunk_overlap": 80,
            "separators": [
                "\n\n",      # 段落
                "\n第",      # 法律条款起始
                "。\n",      # 句末换行
                "。",        # 句号
                "；",        # 分号
                "：",        # 冒号
                "\n",        # 换行
                " ",         # 空格
            ]
        },
        "contract": {
            "chunk_size": 400,
            "chunk_overlap": 60,
            "separators": [
                "\n\n",
                "\n第",
                "\n（",
                "\n(",
                "。\n",
                "。",
                "；",
                "\n",
            ]
        },
        "general": {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "separators": ["\n\n", "\n", "。", "！", "？", " ", ""]
        }
    }
    
    config = configs.get(splitter_type, configs["general"])
    
    return RecursiveCharacterTextSplitter(
        separators=config["separators"],
        chunk_size=chunk_size or config["chunk_size"],
        chunk_overlap=chunk_overlap or config["chunk_overlap"],
        length_function=len,
        is_separator_regex=False,
    )
