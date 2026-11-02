import logging
import os

from dotenv import load_dotenv
from pymilvus import MilvusClient

load_dotenv()
logger = logging.getLogger(__name__)


def get_milvus_client():
    try:
        milvus_client = MilvusClient(uri=os.getenv("MILVUS_URL"))
        return milvus_client
    except Exception as e:
        logger.exception(f"milvus_client客户端初始化失败：{e}")
        raise


def dense_search(collection_name: str, dense_vector, top_k: int = 10,
                 filter_expr: str = None, output_fields: list = None):
    """稠密向量单路检索（COSINE）。"""
    try:
        milvus_client = get_milvus_client()
        search_params = {
            "collection_name": collection_name,
            "data": [dense_vector],
            "anns_field": "dense_vector",
            "search_params": {"metric_type": "COSINE"},
            "limit": top_k,
        }
        if filter_expr:
            search_params["filter"] = filter_expr
        if output_fields:
            search_params["output_fields"] = output_fields

        results = milvus_client.search(**search_params)
        count = len(results[0]) if results else 0
        logger.info(f"稠密检索完成，collection={collection_name}，返回{count}条结果")
        return results[0] if results else []
    except Exception as e:
        logger.exception(f"稠密检索失败：{e}")
        raise


def sparse_search(collection_name: str, sparse_vector, top_k: int = 10,
                  filter_expr: str = None, output_fields: list = None):
    """稀疏向量单路检索（IP，BM25 权重）。"""
    try:
        milvus_client = get_milvus_client()
        search_params = {
            "collection_name": collection_name,
            "data": [sparse_vector],
            "anns_field": "sparse_vector",
            "search_params": {"metric_type": "IP"},
            "limit": top_k,
        }
        if filter_expr:
            search_params["filter"] = filter_expr
        if output_fields:
            search_params["output_fields"] = output_fields

        results = milvus_client.search(**search_params)
        count = len(results[0]) if results else 0
        logger.info(f"稀疏检索完成，collection={collection_name}，返回{count}条结果")
        return results[0] if results else []
    except Exception as e:
        logger.exception(f"稀疏检索失败：{e}")
        raise


def fetch_all_contents(collection_name: str, batch_size: int = 1000) -> list:
    """分页读取 collection 中所有 chunk 的 content 文本（用于 BM25 全量 fit）。"""
    milvus_client = get_milvus_client()
    if not milvus_client.has_collection(collection_name):
        return []
    contents = []
    offset = 0
    while True:
        results = milvus_client.query(
            collection_name=collection_name,
            filter="",
            output_fields=["content"],
            limit=batch_size,
            offset=offset,
        )
        if not results:
            break
        contents.extend((r.get("content") or "") for r in results)
        if len(results) < batch_size:
            break
        offset += len(results)
    return contents
