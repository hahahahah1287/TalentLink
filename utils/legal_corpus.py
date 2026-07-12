# -*- coding: utf-8 -*-
"""Helpers for loading one or more statute text files into article Documents."""
import glob
import hashlib
import os
from typing import Iterable, List

from langchain_core.documents import Document

from utils.legal_parser import parse_legal_document


DEFAULT_LEGAL_SOURCE_GLOBS = [
    "labor_law.txt",
    "data/legal_sources/**/*.txt",
]


def resolve_knowledge_base_paths(retrieval_config) -> List[str]:
    """Resolve configured law-source files while keeping old single-file config compatible."""
    raw_paths: List[str] = []
    configured = getattr(retrieval_config, "knowledge_base_paths", None)
    if configured:
        raw_paths.extend(configured)
    legacy = getattr(retrieval_config, "knowledge_base_path", "")
    if legacy:
        raw_paths.append(legacy)
    if not raw_paths:
        raw_paths = list(DEFAULT_LEGAL_SOURCE_GLOBS)

    resolved: List[str] = []
    seen = set()
    for item in raw_paths:
        matches = glob.glob(item, recursive=True) if any(ch in item for ch in "*?[") else [item]
        for path in matches:
            if not path or not os.path.isfile(path):
                continue
            norm = os.path.normpath(path)
            if norm in seen:
                continue
            seen.add(norm)
            resolved.append(norm)
    return resolved


def parse_legal_corpus(paths: Iterable[str]) -> List[Document]:
    """Parse all configured statute files into article-level Documents."""
    docs: List[Document] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        docs.extend(parse_legal_document(text, source=path))
    return docs


def corpus_fingerprint(paths: Iterable[str]) -> str:
    """Stable short fingerprint for cache/index invalidation diagnostics."""
    h = hashlib.sha1()
    for path in sorted(paths):
        h.update(os.path.normpath(path).encode("utf-8"))
        try:
            stat = os.stat(path)
            h.update(str(int(stat.st_mtime)).encode("utf-8"))
            h.update(str(stat.st_size).encode("utf-8"))
        except OSError:
            h.update(b"missing")
    return h.hexdigest()[:12]
