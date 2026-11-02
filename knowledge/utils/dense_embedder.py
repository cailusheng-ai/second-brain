"""dense 稠密向量客户端

调用私有化部署的 embedding 服务（bge-base-zh-v1.5，768 维），
替代原本地 BGE-M3 的 dense 职责。HTTP 调用统一异步（httpx.AsyncClient）。

用法：
    embedder = get_dense_embedder()
    docs = await embedder.embed_documents(["文档1", "文档2"])
    queries = await embedder.embed_queries(["查询1"])
"""

import logging
import os
from typing import List

import httpx
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class DenseEmbedder:
    """dense 向量 HTTP 客户端。"""

    def __init__(self, base_url: str | None = None):
        self.base_url = (
            base_url
            or os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8002")
        ).rstrip("/")

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """文档编码（不加指令前缀）。"""
        return await self._embed("/embed_documents", texts)

    async def embed_queries(self, texts: List[str]) -> List[List[float]]:
        """查询编码（服务端自动拼接 bge 指令前缀）。"""
        return await self._embed("/embed_queries", texts)

    async def _embed(self, endpoint: str, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # 每次请求独立建 client，避免跨事件循环复用（导入端同步图 / 查询端异步图）
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{self.base_url}{endpoint}", json={"texts": texts}
            )
            resp.raise_for_status()
            return resp.json()["embeddings"]


_dense_embedder: DenseEmbedder | None = None


def get_dense_embedder() -> DenseEmbedder:
    global _dense_embedder
    if _dense_embedder is None:
        _dense_embedder = DenseEmbedder()
    return _dense_embedder
