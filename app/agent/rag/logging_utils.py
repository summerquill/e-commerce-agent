"""RAG 检索日志器：终端 + 文件双通道，供 retriever/reranker/tool 共用。"""

from pathlib import Path
import logging
import sys
from typing import Optional

from app.config.settings import settings

_logger: Optional[logging.Logger] = None


def get_rag_logger() -> logging.Logger:
    """构建 RAG 检索日志器：同时写终端和 settings.rag_log_path。"""
    global _logger
    if _logger is not None:
        return _logger

    logger = logging.getLogger("ecom.rag")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    log_path = Path(settings.rag_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    _logger = logger
    return logger


def reset_rag_logger() -> None:
    """关闭并清空日志器（测试或切换日志路径时使用）。"""
    global _logger
    if _logger is not None:
        for handler in _logger.handlers:
            handler.close()
        _logger.handlers.clear()
    _logger = None
