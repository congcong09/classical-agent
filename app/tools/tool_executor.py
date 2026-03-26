from typing import Any, Callable


class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具
    """

    def __init__(self):
        self.tools: dict[str, Any] = {}

    def register_tool(self, name: str, description: str, func: Callable):
        if name in self.tools:
            print(f"⚠️ 工具 「{name}」已经存在，将被覆盖！")

        self.tools[name] = {"name": name, "description": description, "func": func}

        print(f"😄 工具 「{name}」 已注册")

    def get_tool(self, name: str) -> Callable:
        return self.tools[name].get("func")

    def get_available_tools(self) -> str:
        return "\n".join(
            [f"- {name}: {info['description']}" for name, info in self.tools.items()]
        )


if __name__ == "__main__":
    from .search import search

    tool_executor = ToolExecutor()
    tool_executor.register_tool(
        "search",
        "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。",
        search,
    )
    print(tool_executor.get_available_tools())
    print("=" * 30)

    tool_name = "search"
    tool_input = "苹果有什么新产品"

    tool_func = tool_executor.get_tool(tool_name)
    if tool_func:
        res = tool_func(tool_input)
        print(res)
