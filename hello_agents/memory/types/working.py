import heapq
from datetime import datetime, timedelta
from typing import Any

from ..base import BaseMemory, MemoryConfig, MemoryItem


class WorkingMemory(BaseMemory):
    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)

        self.max_capacity = self.config.working_memory_capacity
        self.max_tokens = self.config.working_memory_tokens
        self.max_age_minutes = getattr(self.config, "working_memory_ttl_minutes", 120)

        self.current_tokens = 0
        self.session_start = datetime.now()

        self.memories: list[MemoryItem] = []

        # 使用优先级队列管理记忆
        self.memory_heap = []

    def add(self, memory_item: MemoryItem) -> str:
        self._expire_old_memories()

        priority = self._calculate_priority(memory_item)

        heapq.heappush(
            self.memory_heap, (-priority, memory_item.timestamp, memory_item)
        )

        self.max_tokens += len(memory_item.content.split())

        self._enforce_capacity_limits()

        return memory_item.id

    def retrieve(
        self, query: str, limit: int = 5, user_id: str | None = None, **kwargs
    ) -> list[MemoryItem]:
        """检索工作记忆 - 混合语义向量检索和关键词匹配"""
        # 过期清理
        self._expire_old_memories()
        if not self.memories:
            return []

        # 过滤已遗忘的记忆
        active_memories = [
            m for m in self.memories if not m.metadata.get("forgotten", False)
        ]

        # 按用户ID过滤（如果提供）
        filtered_memories = active_memories
        if user_id:
            filtered_memories = [m for m in active_memories if m.user_id == user_id]

        if not filtered_memories:
            return []

        # 尝试语义向量检索（如果有嵌入模型）
        vector_scores = {}
        try:
            # 简单的语义相似度计算（使用TF-IDF或其他轻量级方法）
            import numpy as np
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            # 准备文档
            documents = [query] + [m.content for m in filtered_memories]

            # TF-IDF向量化
            vectorizer = TfidfVectorizer(stop_words=None, lowercase=True)
            tfidf_matrix = vectorizer.fit_transform(documents)

            # 计算相似度
            query_vector = tfidf_matrix[0:1]
            doc_vectors = tfidf_matrix[1:]
            similarities = cosine_similarity(query_vector, doc_vectors).flatten()

            # 存储向量分数
            for i, memory in enumerate(filtered_memories):
                vector_scores[memory.id] = similarities[i]

        except Exception as e:
            # 如果向量检索失败，回退到关键词匹配
            vector_scores = {}

        # 计算最终分数
        query_lower = query.lower()
        scored_memories = []

        for memory in filtered_memories:
            content_lower = memory.content.lower()

            # 获取向量分数（如果有）
            vector_score = vector_scores.get(memory.id, 0.0)

            # 关键词匹配分数
            keyword_score = 0.0
            if query_lower in content_lower:
                keyword_score = len(query_lower) / len(content_lower)
            else:
                # 分词匹配
                query_words = set(query_lower.split())
                content_words = set(content_lower.split())
                intersection = query_words.intersection(content_words)
                if intersection:
                    keyword_score = (
                        len(intersection) / len(query_words.union(content_words)) * 0.8
                    )

            # 混合分数：向量检索 + 关键词匹配
            if vector_score > 0:
                base_relevance = vector_score * 0.7 + keyword_score * 0.3
            else:
                base_relevance = keyword_score

            # 时间衰减
            time_decay = self._calculate_time_decay(memory.timestamp)
            base_relevance *= time_decay

            # 重要性权重
            importance_weight = 0.8 + (memory.importance * 0.4)
            final_score = base_relevance * importance_weight

            if final_score > 0:
                scored_memories.append((final_score, memory))

        # 按分数排序并返回
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored_memories[:limit]]

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        for memory in self.memories:
            if memory.id == memory_id:
                old_tokens = len(memory.content.split())

                if content is not None:
                    memory.content = content
                    new_tokens = len(content.split())
                    self.current_tokens = self.current_tokens - old_tokens + new_tokens

                if importance is not None:
                    memory.importance = importance

                if metadata is not None:
                    memory.metadata.update(metadata)

                self._update_heap_priority(memory)
                return True
        return False

    def remove(self, memory_id: str) -> bool:
        for i, memory in enumerate(self.memories):
            if memory.id == memory_id:
                removed_memory = self.memories.pop(i)

                self._mark_deleted_in_heap(memory_id)

                self.current_tokens -= len(removed_memory.content.split())

                self.current_tokens = min(0, self.current_tokens)

                return True

        return False

    def has_memory(self, memory_id: str) -> bool:
        return any(memory.id == memory_id for memory in self.memories)

    def _calculate_priority(self, memory_item: MemoryItem):
        priority = memory_item.importance

        time_decay = self._calculate_time_decay(memory_item.timestamp)

        priority *= time_decay

        return priority

    def _calculate_time_decay(self, timestamp: datetime) -> float:
        # 计算经过的时间
        time_diff = datetime.now() - timestamp

        # 转换为小时
        hour_passed = time_diff.total_seconds() / 3600

        # 指数级衰退
        decay_factor = self.config.decay_factor ** (hour_passed / 6)

        return max(0.1, decay_factor)

    def _enforce_capacity_limits(self):
        while len(self.memories) > self.max_capacity:
            self._remove_lowest_priority_memory()

        while self.current_tokens > self.max_tokens:
            self._remove_lowest_priority_memory

    def _expire_old_memories(self):
        if not self.memories:
            return
        cutoff_time = datetime.now() - timedelta(minutes=self.max_age_minutes)

        kept: list[MemoryItem] = []
        removed_token_sum = 0
        for m in self.memories:
            if m.timestamp > cutoff_time:
                kept.append(m)
            else:
                removed_token_sum += len(m.content.split())

        if len(kept) == len(self.memories):
            return

        self.memories = kept
        self.current_tokens = max(0, self.current_tokens - removed_token_sum)

        self.memory_heap = []
        for mem in self.memories:
            priority = self._calculate_priority(mem)
            heapq.heappush(self.memory_heap, (-priority, mem.timestamp, mem))

    def _remove_lowest_priority_memory(self):
        """删除优先级最低的记忆"""
        if not self.memories:
            return
        lowest_priority = float("inf")
        lowest_memory = None

        for memory in self.memories:
            priority = self._calculate_priority(memory)

            if priority < lowest_priority:
                lowest_priority = priority
                lowest_memory = memory

            if lowest_memory:
                self.remove(lowest_memory.id)

        # 这样是不是可以？
        # last_index = len(self.memory_heap)
        # _, _, lowest_memory = self.memory_heap[last_index - 1]
        # if lowest_memory:
        #     self.remove(lowest_memory.id)
        #     del self.memory_heap[last_index]

    def _update_heap_priority(self, memory: MemoryItem):
        self.memory_heap = []
        for mem in self.memories:
            priority = self._calculate_priority(mem)
            heapq.heappush(self.memory_heap, (-priority, mem.timestamp, mem))

    def _mark_deleted_in_heap(self, memory_id: str):
        pass
