# -*- coding: utf-8 -*-
"""LLM backend factory.

默认仍使用进程内 ChatLlamaCpp；设置 LLM_BACKEND=server 时，应用进程只创建
OpenAI-compatible HTTP client，不再加载 GGUF 模型文件。
"""
import os
from typing import Any


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def get_llm_backend(llm_config: Any) -> str:
    """Resolve LLM backend from env first, then AppConfig."""
    backend = _env("LLM_BACKEND", getattr(llm_config, "backend", "local"))
    return backend.lower()


def llm_supports_parallel_requests(llm_config: Any) -> bool:
    """Whether app-level LLM calls can be submitted concurrently."""
    return get_llm_backend(llm_config) == "server"


def describe_llm_backend(llm_config: Any) -> str:
    backend = get_llm_backend(llm_config)
    if backend == "server":
        base_url = _env("LLM_SERVER_BASE_URL", getattr(llm_config, "server_base_url", "http://127.0.0.1:8000/v1"))
        model = _env("LLM_SERVER_MODEL", getattr(llm_config, "server_model", "local-9b"))
        return f"llama_cpp.server ({model} @ {base_url})"
    return f"ChatLlamaCpp ({getattr(llm_config, 'model_path', '')})"


def create_chat_llm(llm_config: Any, *, streaming: bool):
    """Create the chat model used by generation, HyDE and specialist extraction."""
    backend = get_llm_backend(llm_config)

    if backend == "server":
        from langchain_openai import ChatOpenAI

        base_url = _env("LLM_SERVER_BASE_URL", getattr(llm_config, "server_base_url", "http://127.0.0.1:8000/v1"))
        model = _env("LLM_SERVER_MODEL", getattr(llm_config, "server_model", "local-9b"))
        api_key = _env("LLM_SERVER_API_KEY", getattr(llm_config, "server_api_key", "not-needed"))
        timeout = float(_env("LLM_SERVER_TIMEOUT", str(getattr(llm_config, "server_timeout", 600))))
        return ChatOpenAI(
            model_name=model,
            openai_api_base=base_url,
            openai_api_key=api_key,
            temperature=getattr(llm_config, "temperature", 0.1),
            streaming=streaming,
            request_timeout=timeout,
        )

    if backend == "local":
        try:
            from langchain_experimental.chat_models import ChatLlamaCpp
        except ImportError:
            from langchain_community.chat_models import ChatLlamaCpp

        return ChatLlamaCpp(
            model_path=llm_config.model_path,
            n_gpu_layers=llm_config.n_gpu_layers,
            n_ctx=llm_config.n_ctx,
            temperature=llm_config.temperature,
            verbose=llm_config.verbose,
            streaming=streaming,
        )

    raise ValueError(f"Unsupported LLM_BACKEND: {backend}. Use 'local' or 'server'.")
