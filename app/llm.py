import os
from openai import OpenAI
from typing import Any
from dotenv import load_dotenv
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

load_dotenv()


class HelloAgentsLLM:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
    ):
        self.model = model or os.getenv("LLM_MODEL_ID", "")
        api_key = api_key or os.getenv("LLM_API_KEY", "")
        base_url = base_url or os.getenv("LLM_BASE_URL", "")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        if not all([self.model, api_key, base_url]):
            raise ValueError("模型ID，api密钥和服务地址必须提供或在.env文件中定义")

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def think(
        self, messages: list[ChatCompletionMessageParam], temperature: int = 0
    ) -> str | None:
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            print("✅ 大语言模型响应成功：")
            collected_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()
            return "".join(collected_content)
        except Exception as e:
            print(f"❌ 调用LLM API时发生错误：{e}")


if __name__ == "__main__":
    llm_client = HelloAgentsLLM()

    example_messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": "You are a helpful assistant that writes Python code.",
        },
        {"role": "user", "content": "今天是哪年？几月几号？"},
    ]

    print("--- 调用 ---")
    response_text = llm_client.think(messages=example_messages)

    if response_text:
        print("\n\n--- 模型完整的相应 ---")
        print(response_text)
