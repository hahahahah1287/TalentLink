# -*- coding: utf-8 -*-
"""
索引构建脚本

只运行一次，构建带元数据标注的 FAISS 索引。
后续启动 main.py 会直接加载索引，不再重复构建。

使用：
    python build_index.py
    python build_index.py --skip-metadata

环境变量：
    BUILD_INDEX_SKIP_METADATA=1 python build_index.py
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import AppConfig
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from utils.embeddings import TransformerEmbeddings
from utils.legal_corpus import corpus_fingerprint, parse_legal_corpus, resolve_knowledge_base_paths


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def build(skip_metadata: bool = False):
    config = AppConfig()
    knowledge_paths = resolve_knowledge_base_paths(config.retrieval)
    index_path = config.retrieval.faiss_index_path
    metadata_index_path = index_path + "_with_metadata"
    # 跳过已存在
    if os.path.exists(metadata_index_path):
        print(f"✅ 索引已存在: {metadata_index_path}")
        print(f"   如需重建，请先删除该目录")
        return

    # 检查知识库
    if not knowledge_paths:
        print("❌ 未找到知识库文件；请放入 labor_law.txt 或 data/legal_sources/**/*.txt")
        return

    corpus_id = corpus_fingerprint(knowledge_paths)

    print(f"📦 加载 Embedding...")
    embeddings = TransformerEmbeddings(
        model_name=config.embedding.model_name,
        device=config.embedding.device,
        normalize_embeddings=config.embedding.normalize_embeddings,
        local_files_only=True,
    )

    # 1. 解析法律文档
    print(f"📄 解析法律文档: {knowledge_paths}")
    docs = parse_legal_corpus(knowledge_paths)
    for doc in docs:
        doc.metadata["corpus_version"] = config.retrieval.corpus_version
        doc.metadata["corpus_fingerprint"] = corpus_id
    print(f"✅ 解析为 {len(docs)} 个条款，corpus_fingerprint={corpus_id}")

    if skip_metadata:
        print("⏭️  跳过 LLM 元数据标注，直接使用基础条款 metadata 构建索引")
    else:
        # 2. 加载 LLM（用于元数据标注）
        print(f"📦 加载模型: {config.llm.model_path} ...")
        t0 = time.time()
        from langchain_community.chat_models import ChatLlamaCpp
        from utils import annotate_documents

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


def parse_args():
    parser = argparse.ArgumentParser(description="构建劳动法 FAISS 检索索引")
    parser.add_argument(
        "--skip-metadata",
        "--no-metadata",
        action="store_true",
        help="跳过 LLM 加载与元数据抽取，直接用 bge-m3 对原始条款向量化",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(skip_metadata=args.skip_metadata or _env_flag_enabled("BUILD_INDEX_SKIP_METADATA"))
