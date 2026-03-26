import os
import json
from serpapi import SerpApiClient


def search(query: str) -> str | None:
    """
    一个基于SerpApi的实战网页搜索引擎工具。
    它会智能的解析搜索结果，优先返回直接回答或只是图谱信息。
    """
    print(f"🔍 正在执行 [SerpApi] 网页搜索：{query}")

    try:
        api_key = os.getenv("SERPAPI_API_KEY", "")
        if not api_key:
            raise ValueError(".env 未定义 SERPAPI_API_KEY")

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",
            "hl": "zh-cn",
        }

        client = SerpApiClient(params)
        results = client.get_dict()
        print(json.dumps(results))

        if "organic_results" in results:
            snippets = [
                f"[{i + 1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results.get("organic_results", []))
            ]
            return "\n\n".join(snippets)
        else:
            return f"对不起，没有找到关于「{query}」的信息"

    except Exception as e:
        print(f"❌ 查找 「{query}」 过程中出错 {e}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    res = search("伊朗战争怎么样了？")
    print(res)
