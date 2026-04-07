import os
from datetime import datetime
from typing import Any

from hello_agents.memory.base import MemoryConfig

from ..base import BaseMemory


class Episodic:
    """
    情景记忆中的单个情景
    """

    def __init__(
        self,
        episode_id: str,
        user_id: str,
        session_id: str,
        timestamp: datetime,
        content: str,
        context: dict[str, Any],
        outcome: str | None = None,
        importance: float = 0.5,
    ):
        self.episode = episode_id
        self.user_id = user_id
        self.session_id = session_id
        self.timestamp = timestamp
        self.content = content
        self.context = context
        self.outcome = outcome
        self.importance = importance


class EpisodicMemory(BaseMemory):

    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)

        # 本地缓存（内存）
        self.episodes: list[Episodic] = []
        self.sessions: dict[str, list[str]] = {}  # session_id => episode_id

        # 模式识别缓存
        self.patterns_cache = {}
        self.last_pattern_analysis = None

        db_dir = (
            self.config.storage_path
            if hasattr(self.config, "storage_path")
            else "./memory_data"
        )

        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, "memory.db")
        # self.doc_store =SQL
