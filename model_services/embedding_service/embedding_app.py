"""dense 向量化服务（bge-base-zh-v1.5 私有化部署）

以 FastAPI 形式对外提供 768 维稠密向量，供导入端与查询端调用。

接口：
  POST /embed_documents  文档编码（不加指令前缀）
  POST /embed_queries    查询编码（拼接 bge v1.5 指令前缀）

部署：
  uvicorn main:app --host 0.0.0.0 --port 8002
  模型路径由环境变量 BGE_BASE_ZH_MODEL_PATH 指定。
"""

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# 定位项目根下的 knowledge/.env（含模型路径等配置）
load_dotenv(Path(__file__).resolve().parents[2] / "knowledge" / ".env")

# bge v1.5 系列的查询指令前缀（官方要求，缺失会明显降低召回率）
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


@asynccontextmanager
async def lifespan(app):
    # 启动时预热加载模型，避免首次请求懒加载超时
    await asyncio.to_thread(get_model)
    yield


app = FastAPI(title="embedding-service", version="1.0.0", lifespan=lifespan)

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        model_path = os.getenv("BGE_BASE_ZH_MODEL_PATH")
        if not model_path:
            raise RuntimeError("环境变量 BGE_BASE_ZH_MODEL_PATH 未配置")
        _model = SentenceTransformer(model_path)
    return _model


class EmbedRequest(BaseModel):
    texts: List[str]


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]


@app.post("/embed_documents", response_model=EmbedResponse)
async def embed_documents(req: EmbedRequest) -> EmbedResponse:
    model = get_model()
    embeddings = await asyncio.to_thread(
        model.encode, req.texts, normalize_embeddings=True
    )
    return EmbedResponse(embeddings=[e.tolist() for e in embeddings])


@app.post("/embed_queries", response_model=EmbedResponse)
async def embed_queries(req: EmbedRequest) -> EmbedResponse:
    model = get_model()
    texts = [QUERY_INSTRUCTION + t for t in req.texts]
    embeddings = await asyncio.to_thread(
        model.encode, texts, normalize_embeddings=True
    )
    return EmbedResponse(embeddings=[e.tolist() for e in embeddings])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
