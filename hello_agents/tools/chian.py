from typing import Any

from .registry import ToolRegistry


class ToolChain:
    """工具链 - 支持多个工具的顺序执行"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

        self.steps: list[dict[str, Any]] = []

    def add_step(
        self, tool_name: str, input_template: str, output_key: str | None = None
    ):
        step = {
            "tool_name": tool_name,
            "input_template": input_template,
            "output_key": output_key or f"step_{len(self.steps)}_result",
        }
        self.steps.append(step)
        print(f"✅ 工具链 '{self.name}' 添加步骤: {tool_name}")

    def execute(
        self,
        registry: ToolRegistry,
        input_data: str,
        context: dict[str, Any] | None = None,
    ):
        if not self.steps:
            return "❌ 工具链为空，无法执行"

        print(f"🚀 开始执行工具链: {self.name}")

        if context is None:
            context = {}
        context["input"] = input_data

        final_result = input_data

        for i, step in enumerate(self.steps):
            tool_name = step["tool_name"]
            input_template = step["input_template"]
            output_key = step["output_key"]

            print(f"📝 执行步骤 {i + 1}/{len(self.steps)}: {tool_name}")

            # 替换模板中的变量
            try:
                actual_input = input_template.format(**context)
            except KeyError as e:
                return f"❌ 模板变量替换失败: {e}"

            # 执行工具
            try:
                result = registry.execute_tool(tool_name, actual_input)
                context[output_key] = result
                final_result = result
                print(f"✅ 步骤 {i + 1} 完成")
            except Exception as e:
                return f"❌ 工具'tool_name'执行失败：{e}"

        print(f"🎉 工具链'{self.name}'执行完成")
        return final_result


class ToolChainManager:
    """工具链管理器"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.chains: dict[str, ToolChain] = {}
