"""sparse 稀疏向量化（BM25 + 中文分词，无模型）

用 pymilvus 的 BM25EmbeddingFunction + jieba 中文分词，替代原 BGE-M3 的
learned-sparse。BM25 依赖语料统计（IDF），因此导入端 fit 后持久化到本地
IDF 文件，查询端加载同一份统计，保证两端权重一致。

用法：
    # 导入端
    embedder = SparseEmbedder()
    embedder.fit(corpus)
    embedder.save(idf_path)
    sparse = embedder.encode_documents(texts)

    # 查询端
    embedder = get_sparse_embedder()  # 自动加载 IDF 文件
    q_sparse = embedder.encode_queries([query])
"""

import logging
import os
import pickle
from typing import Dict, List

from dotenv import load_dotenv
from pymilvus.model.sparse import BM25EmbeddingFunction
from pymilvus.model.sparse.bm25.tokenizers import Analyzer, JiebaTokenizer, PunctuationFilter

load_dotenv()
logger = logging.getLogger(__name__)


def _csr_row_to_dict(mat, i: int) -> Dict[int, float]:
    """csr 稀疏矩阵的第 i 行 → {token_id: weight} dict。"""
    start, end = mat.indptr[i], mat.indptr[i + 1]
    return {int(c): float(v) for c, v in zip(mat.indices[start:end], mat.data[start:end])}


# 常见中文标点（PunctuationFilter 额外过滤，避免依赖 nltk 的 StopwordFilter）
_ZH_PUNCTUATION = "，。、；：？！“”‘’（）《》【】…—·"


def _build_zh_analyzer() -> Analyzer:
    """构建中文分词器：jieba 分词 + 去标点（不用 StopwordFilter，避免 nltk 依赖）。

    BM25 的 IDF 会自然压低高频停用词权重，因此去掉停用词过滤对效果影响很小。
    """
    tokenizer = JiebaTokenizer()
    filters = [PunctuationFilter(extras=_ZH_PUNCTUATION)]
    return Analyzer(name="zh", tokenizer=tokenizer, preprocessors=[], filters=filters)


class SparseEmbedder:
    """BM25 稀疏向量化。"""

    def __init__(self):
        self.analyzer = _build_zh_analyzer()
        self.bm25 = BM25EmbeddingFunction(self.analyzer)

    def fit(self, corpus: List[str]) -> None:
        """在语料上拟合，收集文档频率统计（IDF）。"""
        self.bm25.fit(corpus)

    def save(self, idf_path: str) -> None:
        parent = os.path.dirname(idf_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(idf_path, "wb") as f:
            pickle.dump(self.bm25, f)
        logger.info(f"BM25 IDF 统计已保存: {idf_path}")

    @classmethod
    def load(cls, idf_path: str) -> "SparseEmbedder":
        with open(idf_path, "rb") as f:
            bm25 = pickle.load(f)
        obj = cls()
        obj.bm25 = bm25
        return obj

    def encode_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        if not texts:
            return []
        mat = self.bm25.encode_documents(texts)
        return [_csr_row_to_dict(mat, i) for i in range(mat.shape[0])]

    def encode_queries(self, texts: List[str]) -> List[Dict[int, float]]:
        if not texts:
            return []
        mat = self.bm25.encode_queries(texts)
        return [_csr_row_to_dict(mat, i) for i in range(mat.shape[0])]


def get_sparse_embedder() -> SparseEmbedder:
    """获取 SparseEmbedder：存在 IDF 文件则加载，否则返回未拟合的新实例。"""
    idf_path = os.getenv("SPARSE_IDF_PATH")
    if idf_path and os.path.exists(idf_path):
        return SparseEmbedder.load(idf_path)
    return SparseEmbedder()
