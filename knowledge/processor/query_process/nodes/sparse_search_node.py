import asyncio
import os

from dotenv import load_dotenv

from knowledge.processor.query_process.base import BaseNode, build_topic_filter
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.milvus_client_utils import sparse_search
from knowledge.utils.sparse_embedder import get_sparse_embedder

load_dotenv()

_OUTPUT_FIELDS = ["chunk_id", "content", "title", "chapter_path", "file_title", "topics"]


class SparseSearchNode(BaseNode):
    """稀疏(BM25)关键词检索节点。"""

    name = "sparse_search_node"

    async def process(self, state: QueryGraphState) -> QueryGraphState:
        topics = state.get("topics") or []
        if isinstance(topics, str):
            topics = [topics]

        rewritten_query = state.get("rewritten_query")
        if not rewritten_query:
            raise StateFieldError(node_name=self.name, field_name="rewritten_query")

        filter_expr = build_topic_filter(topics)

        # BM25 编码 query（CPU 阻塞，放线程池）
        sparse_embedder = get_sparse_embedder()
        sparse_vector = (await asyncio.to_thread(
            sparse_embedder.encode_queries, [rewritten_query]
        ))[0]

        # sparse 检索（同步 Milvus 放线程池）
        search_result = await asyncio.to_thread(
            sparse_search,
            os.getenv("CHUNK_COLLECTION_NAME"),
            sparse_vector,
            self.config.sparse_search_limit,
            filter_expr,
            _OUTPUT_FIELDS,
        )

        sparse_chunks = []
        for hit in (search_result or []):
            entity = hit.get("entity", {})
            sparse_chunks.append({**entity, "_sparse_distance": hit.get("distance", 0)})

        return {"sparse_chunks": sparse_chunks}
