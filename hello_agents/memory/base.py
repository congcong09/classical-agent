"""
记忆系统基础类和配置

按照第8章架构设计的基础组件：
- MemoryItem: 记忆项数据结构
- MemoryConfig: 记忆系统配置
- BaseMemory: 记忆基类
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class MemoryItem(BaseModel):
    """记忆项数据结构"""

    id: str
    content: str
    memory_type: str
    user_id: str
    timestamp: datetime
    importance: float = 0.5
    metadata: dict[str, Any] = {}

    class Config:
        arbitrary_types_allowed = True


class MemoryConfig(BaseModel):
    """记忆系统配置"""

    # 存储路径
    storage_path: str = "./memory_data"

    # 统计先使用的基础配置
    max_capacity: int = 100
    importance_threshold: float = 0.1
    decay_factor: float = 0.95

    # 工作记忆特定配置
    working_memory_capacity: int = 100
    working_memory_tokens: int = 2000
    working_memory_ttl_minutes: int = 120

    # 感知记忆特定配置
    perceptual_memory_modalities: list[str] = ["text", "image", "audio", "video"]


class BaseMemory(ABC):
    def __init__(self, config: MemoryConfig, storage_backend=None):
        self.config = config
        self.storage = storage_backend
        self.memory_type = self.__class__.__name__.lower().replace("memory", "")

    @abstractmethod
    def add(self, memory_item: MemoryItem) -> str:
        """
        添加记忆项

        Args:
          memory_item: 记忆项对象

        Returns:
          记忆ID
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> list[MemoryItem]:
        """
        检索相关记忆

        Args:
          query: 查询内容
          limit: 返回数量限制
          **kwargs: 其他检索参数

        Returns:
          相关记忆列表
        """
        pass

    @abstractmethod
    def update(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """更新记忆

        Args:
            memory_id: 记忆ID
            content: 新内容
            importance: 新重要性
            metadata: 新元数据

        Returns:
            是否更新成功
        """
        pass

    @abstractmethod
    def remove(self, memory_id: str) -> bool:
        pass

    @abstractmethod
    def has_memory(self, memory_id: str) -> bool:
        pass

    @abstractmethod
    def clear(self) -> bool:
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        pass

    def generate_id(self) -> str:
        import uuid

        return str(uuid.uuid4())

    def _calculate_importance(
        self, content: str, base_importance: float = 0.5
    ) -> float:
        importance = base_importance

        if len(content) > 100:
            importance += 0.1

        important_words = ["重要", "关键", "必须", "注意", "警告", "错误"]

        if any(word in content for word in important_words):
            importance += 0.2

        return max(0.0, min(1.0, importance))

    def __str__(self):
        stats = self.get_stats()
        return f"{self.__class__.__name__}(count={stats.get('count', 0)})"

    def __repr__(self) -> str:
        return self.__str__()
