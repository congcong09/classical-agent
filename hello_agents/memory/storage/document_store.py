"""
文档存储实现

支持多种文档数据库后端
- SQLite: 轻量级关系型数据库
- PostgreSQL: 企业级关系型数据库
"""

import os
import sqlite3
import threading
from abc import ABC, abstractmethod
from typing import Any


class DocumentStore(ABC):
    @abstractmethod
    def add_memory(
        self,
        memory_id: str,
        user_id: str,
        content: str,
        memory_type: str,
        timestamp: int,
        importance: int,
        properties: dict[str, Any] | None = None,
    ):
        """添加记忆"""
        pass

    @abstractmethod
    def get_memory(self, memory_id: str):
        pass

    @abstractmethod
    def search_memories(
        self,
        user_id: str | None = None,
        memory_type: str | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        importance_threshold: float | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        properties: dict[str, Any] | None = None,
    ) -> bool:
        pass

    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        pass

    @abstractmethod
    def get_database_stats(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def add_document(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        pass

    @abstractmethod
    def get_document(self, document_id: str) -> dict[str, Any] | None:
        pass


class SQLiteDocumentStore(DocumentStore):
    _instances = {}  # 存储已经创建的实例
    _initialized_dbs = set()  # 存储已初始化的数据库路径

    def __new__(cls, db_path: str = "./memory.db"):
        """单例模式，同一路径下只创建一个实例"""

        abs_path = os.path.abspath(db_path)
        if abs_path not in cls._instances:
            # 这里是 python 中很特殊的实例创建方式
            instance = super(SQLiteDocumentStore, cls).__new__(cls)
            cls._instances[abs_path] = instance
        return cls._instances[abs_path]

    def __init__(self, db_path: str = "./memory.db"):
        # 避免重复初始化
        if hasattr(self, "_initialized"):
            return

        self.db_path = db_path
        self.local = threading.local()

        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        abs_path = os.path.abspath(db_path)
        if abs_path not in self._initialized_dbs:
            self._init_database()
            self._initialized_dbs.add(abs_path)
            print(f"[OK] SQLite 文档存储初始化完成：{db_path}")

        self._initialized = True

    def _get_connection(self):
        """获取现成本地连接"""
        if not hasattr(self.local, "connection"):
            self.local.connection = sqlite3.connect(self.db_path)
            self.local.connection.row_factory = sqlite3.Row
        return self.local.connection

    def _init_database(self):
        """初始化数据库表"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # 创建用户表
        cursor.execute("""

          """)
