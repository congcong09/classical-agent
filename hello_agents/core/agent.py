from abc import ABC, abstractmethod

from .config import Config
from .llm import HelloAgentsLLM
from .message import Message


class Agent(ABC):
    """Agent基类"""

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: str | None = None,
        config: Config | None = None,
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: list[Message] = []

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """运行agent"""
        pass

    def add_message(self, message: Message):
        self._history.append(message)

    def clear_history(self):
        self._history.clear()

    def get_history(self):
        return self._history.copy()

    def __str__(self):
        return f"Agent(name={self.name}, provider={self.llm.provider})"

    def __repr__(self):
        return self.__str__()
