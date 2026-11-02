"""精排客户端（HTTP）

调用私有化部署的 rerank 服务（bge-reranker-v2-m3），替代原本地 FlagReranker。
HTTP 调用统一异步（httpx.AsyncClient）。

用法：
    scores = await rerank(query, documents)
"""

import logging
import os
from typing import List

import httpx
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


async def rerank(query: str, documents: List[str]) -> List[float]:
    """对 [query, doc] 逐对打分，返回与 documents 同序的分数列表。"""
    if not documents:
        return []

    base_url = os.getenv("RERANK_SERVICE_URL", "http://localhost:8003").rstrip("/")
    # 每次请求独立建 client，避免跨事件循环复用（导入端同步图 / 查询端异步图）
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(
            f"{base_url}/rerank", json={"query": query, "documents": documents}
        )
        resp.raise_for_status()
        return resp.json()["scores"]
