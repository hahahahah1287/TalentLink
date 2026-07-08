# -*- coding: utf-8 -*-
"""
Workflow Checkpoint Cache — Redis 缓存机制

架构：
- 将 workflow 的最终结果缓存到 Redis
- 相同 query + scene 命中缓存时直接返回，跳过整个 workflow 执行
- 支持 TTL 过期、手动失效、命中率统计

缓存键设计：
- Key: `checkpoint:{workflow}:{query_hash}`
- Value: JSON 序列化的 AppState 核心字段
- TTL: 可配置，默认 1 小时

为什么用结果缓存而非 LangGraph Checkpoint：
1. LangGraph 的 checkpointer 用于"中断恢复"（human-in-the-loop），
   不适合"相同输入直接返回"的缓存场景
2. 结果缓存粒度更粗但命中率更高：相同问题无论中间状态如何，
   只要最终答案一致就可以复用
3. 法律知识变更频率低，缓存有效期可以设得较长
"""
import hashlib
import json
import time
from typing import Optional, Dict, Any

import redis


class WorkflowCheckpoint:
    """
    Workflow 结果缓存

    基于 Redis 的 checkpoint 缓存，避免重复执行相同查询。

    Usage:
        checkpoint = WorkflowCheckpoint(redis_client)

        # 检查缓存
        cached = checkpoint.get("legal", query)
        if cached:
            return cached

        # 执行 workflow ...
        result = await graph.ainvoke(state)

        # 写入缓存
        checkpoint.set("legal", query, result)
    """

    # 缓存 key 前缀
    KEY_PREFIX = "checkpoint"

    # 默认 TTL（秒）
    DEFAULT_TTL = 3600  # 1 小时

    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        ttl: int = DEFAULT_TTL,
        enabled: bool = True,
    ):
        """
        Args:
            redis_client: Redis 客户端实例（可选，None 时降级为无缓存）
            ttl: 缓存过期时间（秒）
            enabled: 是否启用缓存
        """
        self.redis = redis_client
        self.ttl = ttl
        self.enabled = enabled and redis_client is not None

        # 命中率统计
        self._hits = 0
        self._misses = 0

    def _make_key(self, workflow: str, query: Any) -> str:
        """生成缓存键；query 可为原始字符串或结构化 request fingerprint。"""
        if isinstance(query, dict):
            key_payload = json.dumps(query, ensure_ascii=False, sort_keys=True)
        else:
            key_payload = str(query)
        query_hash = hashlib.sha256(key_payload.encode("utf-8")).hexdigest()[:24]
        return f"{self.KEY_PREFIX}:{workflow}:{query_hash}"

    def get(self, workflow: str, query: str) -> Optional[Dict[str, Any]]:
        """
        查询缓存

        Args:
            workflow: 工作流类型（统一为 "legal"）
            query: 用户查询

        Returns:
            缓存的 result dict，未命中返回 None
        """
        if not self.enabled:
            return None

        key = self._make_key(workflow, query)
        try:
            raw = self.redis.get(key)
            if raw:
                self._hits += 1
                data = json.loads(raw)
                print(f"🎯 [Checkpoint] 缓存命中: {workflow} (hits={self._hits})")
                return data
        except Exception as e:
            print(f"⚠️ [Checkpoint] Redis 读取失败: {e}")

        self._misses += 1
        return None

    def set(
        self,
        workflow: str,
        query: str,
        result: Dict[str, Any],
        ttl: Optional[int] = None,
    ):
        """
        写入缓存

        只缓存成功的结果（final_answer 非空且长度 > 10）。

        Args:
            workflow: 工作流类型
            query: 用户查询
            result: workflow 执行结果（AppState dict）
            ttl: 覆盖默认 TTL（可选）
        """
        if not self.enabled:
            return

        # 只缓存有效结果
        final_answer = result.get("final_answer", "")
        if not final_answer or len(final_answer.strip()) < 10:
            return

        # 提取需要缓存的字段（不缓存中间状态）
        cache_data = {
            "final_answer": final_answer,
            "tool_history": result.get("tool_history", []),
            "law_context": result.get("law_context", ""),
            "evidence_items": result.get("evidence_items", []),
            "cached_at": time.time(),
        }

        key = self._make_key(workflow, query)
        effective_ttl = ttl or self.ttl

        try:
            self.redis.setex(
                key,
                effective_ttl,
                json.dumps(cache_data, ensure_ascii=False),
            )
            print(f"💾 [Checkpoint] 缓存写入: {workflow} (ttl={effective_ttl}s)")
        except Exception as e:
            print(f"⚠️ [Checkpoint] Redis 写入失败: {e}")

    def invalidate(self, workflow: str, query: str):
        """手动失效某条缓存"""
        if not self.enabled:
            return
        key = self._make_key(workflow, query)
        try:
            self.redis.delete(key)
        except Exception:
            pass

    def invalidate_all(self, workflow: Optional[str] = None):
        """
        批量失效缓存

        Args:
            workflow: 指定工作流类型（None 时清除所有）
        """
        if not self.enabled:
            return
        pattern = f"{self.KEY_PREFIX}:{workflow}:*" if workflow else f"{self.KEY_PREFIX}:*"
        try:
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
                print(f"🗑️ [Checkpoint] 已清除 {len(keys)} 条缓存")
        except Exception as e:
            print(f"⚠️ [Checkpoint] 批量清除失败: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """获取缓存指标"""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "enabled": self.enabled,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
            "ttl": self.ttl,
        }
