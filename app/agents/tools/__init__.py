from .search import search
from .tool_executor import ToolExecutor

global_tool_executor = ToolExecutor()

global_tool_executor.register_tool(
    "search",
    "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。",
    search,
)

__all__ = ["global_tool_executor"]
