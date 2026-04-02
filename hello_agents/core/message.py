from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    content: str
    role: MessageRole
    timestamp: datetime | None = None
    metadata: dict[str, Any] | None = None

    def __init__(self, content: str, role: MessageRole, **kwargs):
        super().__init__(
            content=content,
            role=role,
            timestamp=kwargs.get("timestamp", datetime.now()),
            metadata=kwargs.get("metadata", {}),
        )

    def to_dict(self):
        """转换为字典格式（OpenAI API格式）"""
        return {"role": self.role, "content": self.content}

    def __str__(self):
        return f"[{self.role}] {self.content}"
