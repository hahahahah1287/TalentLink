# -*- coding: utf-8 -*-
"""
语义缓存 (Semantic Cache)

基于 Embedding 向量余弦相似度的查询级智能缓存。

传统缓存（如 Redis）基于字符串精确匹配：
  "试用期工资怎么算" ≠ "试用期间的薪资如何计算"  → 缓存未命中

语义缓存基于向量空间距离：
  embed("试用期工资怎么算") ≈ embed("试用期间薪资如何计算")  → 缓存命中

数据流:
  Query → Embedding → 与 Redis 中缓存的向量比较 → 命中则直返回 → 未命中则穿透到 LLM

存储结构 (Redis):
  - Hash: semantic_cache:{hash} → {"query": ..., "response": ..., "embedding": ..., "ts": ...}
  - 使用 SCAN 遍历进行相似度比较（适用于中小规模缓存）
"""
import json
import time
import hashlib
import numpy as np
from typing import Optional, Tuple, List


class SemanticCache:
    """
    基于向量相似度的语义缓存

    使用示例:
        cache = SemanticCache(embeddings=embedding_model, redis_client=redis)

        # 查询
        hit, response = cache.get("试用期工资怎么计算")
        if hit:
            return response

        # 存储
        response = llm.invoke(query)
        cache.put(query, response)
    """

    # Redis key 前缀
    CACHE_PREFIX = "semantic_cache:"

    def __init__(
        self,
        embeddings,
        redis_client=None,
        similarity_threshold: float = 0.93,
        ttl: int = 3600,
        max_cache_size: int = 500,
    ):
        """
        Args:
            embeddings: Embedding 模型实例（需有 embed_query 方法）
            redis_client: Redis 连接实例（可选，无则降级为内存缓存）
            similarity_threshold: 余弦相似度阈值，超过此值判定为命中
            ttl: 缓存过期时间（秒），默认 1 小时
            max_cache_size: 最大缓存条数
        """
        self.embeddings = embeddings
        self.redis = redis_client
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl
        self.max_cache_size = max_cache_size

        # 统计指标
        self.total_queries = 0
        self.cache_hits = 0

        # 降级：内存缓存
        self._memory_cache: List[dict] = []

        # 判断 Redis 是否可用
        self.use_redis = False
        if self.redis:
            try:
                self.redis.ping()
                self.use_redis = True
            except Exception:
                print("⚠️ [SemanticCache] Redis 不可用，降级为内存缓存")

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """计算两个向量的余弦相似度"""
        dot = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def _make_key(self, query: str) -> str:
        """生成缓存 key"""
        digest = hashlib.md5(query.encode("utf-8")).hexdigest()[:12]
        return f"{self.CACHE_PREFIX}{digest}"

    def get(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        查询语义缓存

        Args:
            query: 用户查询

        Returns:
            (命中与否, 缓存的响应内容 or None)
        """
        self.total_queries += 1

        try:
            query_embedding = np.array(
                self.embeddings.embed_query(query), dtype=np.float32
            )
        except Exception as e:
            print(f"⚠️ [SemanticCache] Embedding 失败: {e}")
            return False, None

        best_similarity = 0.0
        best_response = None

        if self.use_redis:
            # 遍历 Redis 中的缓存条目
            cursor = 0
            while True:
                cursor, keys = self.redis.scan(
                    cursor=cursor,
                    match=f"{self.CACHE_PREFIX}*",
                    count=50
                )
                for key in keys:
                    try:
                        data = self.redis.hgetall(key)
                        if not data:
                            continue
                        cached_embedding = np.array(
                            json.loads(data.get("embedding", "[]")),
                            dtype=np.float32
                        )
                        sim = self._cosine_similarity(query_embedding, cached_embedding)
                        if sim > best_similarity:
                            best_similarity = sim
                            best_response = data.get("response", "")
                    except Exception:
                        continue
                if cursor == 0:
                    break
        else:
            # 内存缓存遍历
            for entry in self._memory_cache:
                cached_embedding = np.array(entry["embedding"], dtype=np.float32)
                sim = self._cosine_similarity(query_embedding, cached_embedding)
                if sim > best_similarity:
                    best_similarity = sim
                    best_response = entry["response"]

        if best_similarity >= self.similarity_threshold:
            self.cache_hits += 1
            print(
                f"🎯 [SemanticCache] 缓存命中！相似度: {best_similarity:.4f} "
                f"(阈值: {self.similarity_threshold})"
            )
            return True, best_response

        return False, None

    def put(self, query: str, response: str) -> None:
        """
        写入缓存

        Args:
            query: 用户查询
            response: LLM 响应
        """
        try:
            embedding = self.embeddings.embed_query(query)
        except Exception as e:
            print(f"⚠️ [SemanticCache] Embedding 失败，跳过缓存: {e}")
            return

        cache_data = {
            "query": query,
            "response": response,
            "embedding": json.dumps(embedding),
            "ts": str(time.time()),
        }

        if self.use_redis:
            key = self._make_key(query)
            try:
                self.redis.hset(key, mapping=cache_data)
                self.redis.expire(key, self.ttl)
            except Exception as e:
                print(f"⚠️ [SemanticCache] Redis 写入失败: {e}")
        else:
            # 内存缓存 — LRU 淘汰
            if len(self._memory_cache) >= self.max_cache_size:
                self._memory_cache.pop(0)
            cache_data["embedding"] = embedding  # 内存存原始列表更高效
            self._memory_cache.append(cache_data)

    def get_metrics(self) -> dict:
        """获取缓存命中率等统计指标"""
        hit_rate = (
            self.cache_hits / self.total_queries
            if self.total_queries > 0
            else 0.0
        )
        return {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "hit_rate": f"{hit_rate:.2%}",
            "similarity_threshold": self.similarity_threshold,
            "backend": "redis" if self.use_redis else "memory",
        }
