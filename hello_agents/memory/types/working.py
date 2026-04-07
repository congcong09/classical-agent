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

    def clear(self):
        self.memories = []
        self.memory_heap = []
        self.max_tokens = 0

    def get_stats(self) -> dict[str, Any]:
        self._expire_old_memories()

        active_memories = self.memories

        return {
            "count": len(active_memories),  # 活跃记忆数量
            "forgotten_count": 0,  # 工作记忆中已遗忘的记忆会被直接删除
            "total_count": len(self.memories),  # 总记忆数量
            "current_tokens": self.current_tokens,
            "max_capacity": self.max_capacity,
            "max_tokens": self.max_tokens,
            "max_age_minutes": self.max_age_minutes,
            "session_duration_minutes": (
                datetime.now() - self.session_start
            ).total_seconds()
            / 60,
            "avg_importance": (
                sum(m.importance for m in active_memories) / len(active_memories)
                if active_memories
                else 0.0
            ),
            "capacity_usage": (
                len(active_memories) / self.max_capacity
                if self.max_capacity > 0
                else 0.0
            ),
            "token_usage": (
                self.current_tokens / self.max_tokens if self.max_tokens > 0 else 0.0
            ),
            "memory_type": "working",
        }

    def get_recent_memory(self, limit: int = 10) -> list[MemoryItem]:
        sorted_memories = sorted(self.memories, key=lambda m: m.timestamp, reverse=True)
        return sorted_memories[:limit]

    def get_important(self, limit: int = 10) -> list[MemoryItem]:
        sorted_memories = sorted(
            self.memories, key=lambda m: m.importance, reverse=True
        )
        return sorted_memories[:limit]

    def get_all(self) -> list[MemoryItem]:
        return self.memories.copy()

    def get_context_summary(self, max_length: int = 50) -> str:
        """获取上下文摘要"""
        if not self.memories:
            return "No working memories available"

        sorted_memories = sorted(
            self.memories, key=lambda x: (x.importance, x.timestamp), reverse=True
        )

        summary_parts = []
        current_length = 0

        for memory in sorted_memories:
            content = memory.content

            if current_length + len(content) <= max_length:
                summary_parts.append(content)
                current_length += len(content)
            else:
                remaining = max_length - current_length
                if remaining > 50:
                    summary_parts.append(f"{content[:remaining]}...")
                break

        return "Working Memory Context:\n" + "\n".join(summary_parts)

    def forget(
        self,
        strategy: str = "importance_based",
        threshold: float = 0.1,
        max_age_days: int = 1,
    ) -> int:
        """工作记忆遗忘机制"""
        forgotten_count = 0
        current_time = datetime.now()
        to_remove = []

        # 现将不符合时间要的移除
        cutoff_ttl = current_time - timedelta(minutes=self.max_age_minutes)
        for memory in self.memories:
            if memory.timestamp < cutoff_ttl:
                to_remove.append(memory.id)

        # 根据不同的策略处理
        if strategy == "importance_based":
            for memory in self.memories:
                if memory.importance < threshold:
                    to_remove.append(memory.id)
        elif strategy == "time_based":
            cutoff_ttl = current_time - timedelta(hours=max_age_days * 24)
            for memory in self.memories:
                if memory.timestamp < cutoff_ttl:
                    to_remove.append(memory.id)
        elif strategy == "capacity_based":
            if len(self.memories) > self.max_capacity:
                sorted_memories = sorted(
                    self.memories,
                    key=lambda m: (m.importance, m.timestamp),
                    reverse=True,
                )

                removed_memories = sorted_memories[self.max_capacity :]

                to_remove.extend(m.id for m in removed_memories)
                
        for memory_id in to_remove:
            if self.remove(memory_id):
                forgotten_count += 1

        return forgotten_count

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

    def _update_heap_priority(self, memory: MemoryItem):
        self.memory_heap = []
        for mem in self.memories:
            priority = self._calculate_priority(mem)
            heapq.heappush(self.memory_heap, (-priority, mem.timestamp, mem))

    def _mark_deleted_in_heap(self, memory_id: str):
        pass
