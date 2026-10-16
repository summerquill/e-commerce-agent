"""Checkpointer 工厂：按配置选择持久化后端。

依赖 LangGraph 的 BaseCheckpointSaver 抽象，dev/prod 切换零成本——
不自造抽象层，直接复用框架基类，升级时只换工厂返回值。
"""

from langgraph.checkpoint.base import BaseCheckpointSaver

from app.config.settings import settings


def build_checkpointer() -> BaseCheckpointSaver:
    """根据 settings.checkpointer_backend 返回 checkpointer 实现。"""
    backend = settings.checkpointer_backend

    if backend == "postgres":
        # v2: 生产级持久化，支持中断后跨进程恢复
        # from langgraph.checkpoint.postgres import PostgresSaver
        # return PostgresSaver.from_conn_string(settings.pg_dsn)
        raise NotImplementedError("PostgresSaver 待 v2 接入")

    # 默认 MemorySaver（内存，进程重启即丢，开发期够用）
    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()
