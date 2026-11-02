import asyncio
import os

from dotenv import load_dotenv

from knowledge.processor.query_process.base import BaseNode, build_topic_filter
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.dense_embedder import get_dense_embedder
from knowledge.utils.milvus_client_utils import dense_search

load_dotenv()

_OUTPUT_FIELDS = ["chunk_id", "content", "title", "chapter_path", "file_title", "topics"]


class DenseSearchNode(BaseNode):
    """稠密向量检索节点。"""

    name = "dense_search_node"

    async def process(self, state: QueryGraphState) -> QueryGraphState:
        topics = state.get("topics") or []
        if isinstance(topics, str):
            topics = [topics]

        rewritten_query = state.get("rewritten_query")
        if not rewritten_query:
            raise StateFieldError(node_name=self.name, field_name="rewritten_query")

        filter_expr = build_topic_filter(topics)

        # dense 向量化（HTTP 异步，直接对重写后的 query 编码）
        dense_embedder = get_dense_embedder()
        dense_vector = (await dense_embedder.embed_queries([rewritten_query]))[0]

        # dense 检索（同步 Milvus 放线程池）
        search_result = await asyncio.to_thread(
            dense_search,
            os.getenv("CHUNK_COLLECTION_NAME"),
            dense_vector,
            self.config.dense_search_limit,
            filter_expr,
            _OUTPUT_FIELDS,
        )

        dense_chunks = []
        for hit in (search_result or []):
            entity = hit.get("entity", {})
            dense_chunks.append({**entity, "_dense_distance": hit.get("distance", 0)})

        return {"dense_chunks": dense_chunks}
