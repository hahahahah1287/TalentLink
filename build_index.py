# -*- coding: utf-8 -*-
"""
索引构建脚本

只运行一次，构建带元数据标注的 FAISS 索引。
后续启动 main.py 会直接加载索引，不再重复构建。

使用：
    python build_index.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import AppConfig
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.documents import Document

from utils import parse_legal_document, annotate_documents


def build():
    config = AppConfig()
    knowledge_path = config.retrieval.knowledge_base_path
    index_path = config.retrieval.faiss_index_path
    metadata_index_path = index_path + "_with_metadata"

    print(f"📦 加载 Embedding...")
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=config.embedding.model_name,
        model_kwargs = {
            'device': config.embedding.device,
            "local_files_only": True
        },
        encode_kwargs={'normalize_embeddings': config.embedding.normalize_embeddings},
    )


    # 跳过已存在
    if os.path.exists(metadata_index_path):
        print(f"✅ 索引已存在: {metadata_index_path}")
        print(f"   如需重建，请先删除该目录")
        return

    # 检查知识库
    if not os.path.exists(knowledge_path):
        print(f"❌ 知识库文件不存在: {knowledge_path}")
        return

    # 1. 解析法律文档
    print(f"📄 解析法律文档: {knowledge_path}")
    with open(knowledge_path, 'r', encoding='utf-8') as f:
        text = f.read()
    docs = parse_legal_document(text, source=knowledge_path)
    print(f"✅ 解析为 {len(docs)} 个条款")

    # 2. 加载 LLM（用于元数据标注）
    print(f"📦 加载模型: {config.llm.model_path} ...")
    t0 = time.time()
    llm = ChatLlamaCpp(
        model_path=config.llm.model_path,
        n_gpu_layers=config.llm.n_gpu_layers,
        n_ctx=config.llm.n_ctx,
        temperature=config.llm.temperature,
        verbose=False,
        streaming=False,
    )
    print(f"✅ 模型加载完成 ({time.time() - t0:.1f}s)")

    # 3. 元数据标注
    print(f"🤖 开始元数据标注（共 {len(docs)} 个条款）...")
    t0 = time.time()
    docs = annotate_documents(docs, llm)
    print(f"✅ 标注完成 ({time.time() - t0:.1f}s)")

    # 4. 加载 Embedding
   

    # 5. 构建 FAISS 索引
    print(f"🔨 构建 FAISS 索引...")
    t0 = time.time()
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(metadata_index_path)
    print(f"✅ 索引已保存: {metadata_index_path} ({time.time() - t0:.1f}s)")

    # 6. 验证
    loaded = FAISS.load_local(
        metadata_index_path, embeddings,
        allow_dangerous_deserialization=True
    )
    loaded_docs = list(loaded.docstore._dict.values())
    print(f"✅ 验证通过: 从索引恢复 {len(loaded_docs)} 个条款")

    print(f"\n🎉 索引构建完成！现在可以启动 main.py 了")


if __name__ == "__main__":
    build()
