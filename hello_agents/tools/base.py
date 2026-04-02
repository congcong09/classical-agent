from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ToolParameter(BaseModel):
    """
    工具参数定义
    """

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


class Tool(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, parameters: dict[str, Any]) -> str:
        """执行工具"""
        pass

    @abstractmethod
    def get_parameters(self) -> list[ToolParameter]:
        """获取工具参数定义"""
        pass

    def validate_parameters(self, parameters: dict[str, Any]) -> bool:
        """验证参数"""
        required_params = [p.name for p in self.get_parameters() if p.required]
        return all(param in parameters for param in required_params)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [param.model_dump() for param in self.get_parameters()],
        }

    def to_openai_schema(self):
        parameters = self.get_parameters()

        properties: dict[str, Any] = {}
        required = []

        for param in parameters:
            prop: dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }

            if param.default is not None:
                prop["description"] = f"{param.description}（默认：{param.default}）"

            if param.type == "array":
                prop["items"] = {"type": "string"}  # 默认是字符串数组

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def __str__(self) -> str:
        return f"Tool(name={self.name})"

    def __repr__(self) -> str:
        return self.__str__()
