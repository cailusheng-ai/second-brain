import asyncio
import os

from dotenv import load_dotenv

from knowledge.processor.import_process.base import BaseNode, T
from knowledge.processor.import_process.exceptions import DocumentSplitError, EmbeddingError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.dense_embedder import get_dense_embedder
from knowledge.utils.milvus_client_utils import fetch_all_contents
from knowledge.utils.sparse_embedder import SparseEmbedder

load_dotenv()


class ChunksEmbeddingNode(BaseNode):
    name = "chunks_embedding_and_save_milvus_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        chunks = state.get("chunks", [])
        if not isinstance(chunks, list) or not chunks:
            raise DocumentSplitError("文本加载失败，chunks切分异常，请检查", self.name)
        self.log_step("step1", f"开始对{len(chunks)}条切片进行向量化")

        # 构造嵌入文本：topics + chapter_path + parent_title + title + body
        texts = [
            (doc.get("topics", "") + "\n" +
             doc.get("chapter_path", "") + "\n" +
             doc.get("parent_title", "") + "\n" +
             doc.get("title", "") + "\n" +
             doc.get("body", ""))
            for doc in chunks
        ]

        # dense：HTTP 异步批量向量化（导入端为同步图，用 asyncio.run 包装）
        try:
            dense_vectors = asyncio.run(self._embed_dense(texts))
        except Exception as e:
            raise EmbeddingError(f"dense 向量化失败：{e}", self.name)

        # sparse：BM25 全量 fit（已有语料 + 本次文本）+ 编码本次 chunks
        try:
            sparse_embedder = self._fit_sparse(texts)
            sparse_vectors = sparse_embedder.encode_documents(texts)
        except Exception as e:
            raise EmbeddingError(f"sparse(BM25) 向量化失败：{e}", self.name)

        if len(dense_vectors) != len(chunks) or len(sparse_vectors) != len(chunks):
            raise EmbeddingError(
                f"向量数量不匹配：chunks={len(chunks)}, dense={len(dense_vectors)}, "
                f"sparse={len(sparse_vectors)}",
                self.name,
            )

        out_put = []
        for chunk, dense_vector, sparse_vector in zip(chunks, dense_vectors, sparse_vectors):
            new_chunk = {
                "content": chunk.get("body"),
                "title": chunk.get("title"),
                "parent_title": chunk.get("parent_title", ""),
                "chapter_path": chunk.get("chapter_path", ""),
                "file_title": chunk.get("file_title"),
                "topics": chunk.get("topics"),
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector,
            }
            out_put.append(new_chunk)

        self.log_step("step2", f"已对{len(chunks)}条切片完成向量化")
        state["chunks"] = out_put
        return state

    async def _embed_dense(self, texts):
        """分批异步调用 dense 服务。"""
        dense_embedder = get_dense_embedder()
        dense_vectors = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            dense_vectors.extend(await dense_embedder.embed_documents(batch))
        return dense_vectors

    def _fit_sparse(self, texts) -> SparseEmbedder:
        """BM25 全量 fit：库中已有 content + 本次 texts，并持久化 IDF。"""
        sparse_embedder = SparseEmbedder()
        collection_name = os.getenv("CHUNK_COLLECTION_NAME")
        existing = fetch_all_contents(collection_name) if collection_name else []
        corpus = existing + texts
        sparse_embedder.fit(corpus)

        idf_path = os.getenv("SPARSE_IDF_PATH")
        if idf_path:
            sparse_embedder.save(idf_path)
        return sparse_embedder
