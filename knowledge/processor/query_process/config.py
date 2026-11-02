"""查询流程配置管理模块

集中管理所有配置项，支持环境变量覆盖。所有属性均采用懒加载模式。
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os

from dotenv import load_dotenv
from knowledge.topic_taxonomy import TOPIC_TAXONOMY

load_dotenv()


@dataclass
class QueryConfig:
    """查询流程配置。"""

    # ==================== 文本处理配置 ====================
    max_context_chars: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
    )

    # ==================== 主题标签配置 ====================
    topic_taxonomy: List[str] = field(
        default_factory=lambda: list(TOPIC_TAXONOMY)
    )

    # ==================== Rerank 配置 ====================
    rerank_max_top_k: int = field(
        default_factory=lambda: int(os.getenv("RERANK_MAX_TOP_K", "10"))
    )
    rerank_min_top_k: int = field(
        default_factory=lambda: int(os.getenv("RERANK_MIN_TOP_K", "3"))
    )
    rerank_gap_ratio: float = field(
        default_factory=lambda: float(os.getenv("RERANK_GAP_RATIO", "0.25"))
    )
    rerank_gap_abs: float = field(
        default_factory=lambda: float(os.getenv("RERANK_GAP_ABS", "0.5"))
    )

    # ==================== RRF 配置 ====================
    rrf_k: int = field(
        default_factory=lambda: int(os.getenv("RRF_K", "60"))
    )
    rrf_max_results: int = field(
        default_factory=lambda: int(os.getenv("RRF_MAX_RESULTS", "10"))
    )

    # ==================== 检索配置 ====================
    dense_search_limit: int = field(
        default_factory=lambda: int(os.getenv("DENSE_SEARCH_LIMIT", "10"))
    )
    sparse_search_limit: int = field(
        default_factory=lambda: int(os.getenv("SPARSE_SEARCH_LIMIT", "10"))
    )

    # ==================== LLM 配置 ====================
    openai_api_base: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_BASE", "")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    default_model: str = field(
        default_factory=lambda: os.getenv("MODEL", "")
    )

    # ==================== Milvus 配置 ====================
    milvus_url: str = field(
        default_factory=lambda: os.getenv("MILVUS_URL", "")
    )

    # ==================== MCP 配置 ====================
    mcp_dashscope_base_url: str = field(
        default_factory=lambda: os.getenv("MCP_DASHSCOPE_BASE_URL", "")
    )

    mcp_dashscope_api_key: str = field(
        default_factory=lambda: os.getenv("MCP_DASHSCOPE_API_KEY", "")
    )

    # ==================== 模型服务配置 ====================
    embedding_service_url: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8002")
    )
    rerank_service_url: str = field(
        default_factory=lambda: os.getenv("RERANK_SERVICE_URL", "http://localhost:8003")
    )
    embedding_dim: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "768"))
    )
    sparse_idf_path: str = field(
        default_factory=lambda: os.getenv("SPARSE_IDF_PATH", "")
    )

    @classmethod
    def from_env(cls) -> "QueryConfig":
        """从环境变量加载配置。

        Returns:
            配置实例。
        """
        return cls()


_config: Optional[QueryConfig] = None


def get_config() -> QueryConfig:
    """获取配置单例。

    Returns:
        全局配置实例。
    """
    global _config
    if _config is None:
        _config = QueryConfig.from_env()
    return _config
