# -*- coding: utf-8 -*-
"""
LangGraph 持久化 Checkpointer — 基于 Redis

功能：
- workflow 执行到每个节点后自动保存状态快照
- 进程中断后可从最后一个完成的节点恢复执行
- 支持 TTL 自动清理过期 checkpoint

与 WorkflowCheckpoint（结果缓存）的区别：
- WorkflowCheckpoint：相同 query 直接返回最终结果（业务层缓存）
- LangGraphCheckpoint：中断后从断点恢复（状态机层持久化）

Usage:
    checkpointer = create_langgraph_checkpointer(redis_client)
    graph = workflow.compile(checkpointer=checkpointer)

    # 恢复执行
    config = {"configurable": {"thread_id": "user_123"}}
    state = graph.get_state(config)
    if state.next:  # 有未执行的节点
        result = graph.invoke(None, config=config)  # 从断点继续
"""
import redis
from langgraph.checkpoint.redis import RedisSaver


# checkpoint key 前缀，与 WorkflowCheckpoint 区分
CHECKPOINT_PREFIX = "lg_checkpoint"

# 默认 TTL：2 小时（workflow 不太可能中断超过 2 小时还恢复）
DEFAULT_TTL = 7200


def create_langgraph_checkpointer(
    redis_client: redis.Redis,
    ttl: int = DEFAULT_TTL,
) -> RedisSaver:
    """
    创建基于 Redis 的 LangGraph Checkpointer

    Args:
        redis_client: Redis 客户端实例
        ttl: checkpoint 过期时间（秒）

    Returns:
        RedisSaver 实例，可直接传给 graph.compile(checkpointer=...)
    """
    return RedisSaver(
        redis_client,
        ttl=ttl,
    )


def get_checkpoint_state(graph, thread_id: str) -> dict:
    """
    获取指定 thread 的 checkpoint 状态

    Args:
        graph: 编译好的 LangGraph graph
        thread_id: 会话 ID

    Returns:
        {
            "has_checkpoint": bool,     # 是否有 checkpoint
            "next_nodes": tuple,        # 下一步要执行的节点（空 = 已完成）
            "state": dict or None,      # checkpoint 中的状态
        }
    """
    config = {"configurable": {"thread_id": thread_id}}

    try:
        state = graph.get_state(config)
        return {
            "has_checkpoint": state is not None and len(state.next) > 0,
            "next_nodes": state.next if state else (),
            "state": state.values if state else None,
        }
    except Exception:
        return {
            "has_checkpoint": False,
            "next_nodes": (),
            "state": None,
        }


def resume_from_checkpoint(graph, thread_id: str):
    """
    从断点恢复执行

    Args:
        graph: 编译好的 LangGraph graph
        thread_id: 会话 ID

    Returns:
        执行结果（AppState dict），如果没有 checkpoint 则返回 None
    """
    config = {"configurable": {"thread_id": thread_id}}

    try:
        state = graph.get_state(config)
        if not state or not state.next:
            return None  # 没有断点，不需要恢复

        print(f"🔄 [Checkpoint] 从断点恢复: next={state.next}, thread={thread_id}")

        # invoke(None) = 不传新输入，用 checkpoint 中的状态继续执行
        result = graph.invoke(None, config=config)
        return result

    except Exception as e:
        print(f"⚠️ [Checkpoint] 恢复失败: {e}")
        return None
