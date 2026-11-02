"""精排服务（bge-reranker-v2-m3 私有化部署）

以 FastAPI 形式对外提供 Cross-Encoder 精排打分，供查询端 Reranker 节点调用。

接口：
  POST /rerank  输入 query + documents，返回逐条相关性分数。

部署：
  uvicorn main:app --host 0.0.0.0 --port 8003
  模型路径由环境变量 BGE_RERANKER_MODEL_PATH 指定。
"""

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from FlagEmbedding import FlagReranker

# 定位项目根下的 knowledge/.env（含模型路径等配置）
load_dotenv(Path(__file__).resolve().parents[2] / "knowledge" / ".env")


@asynccontextmanager
async def lifespan(app):
    # 启动时预热加载模型，避免首次请求懒加载超时
    await asyncio.to_thread(get_model)
    yield


app = FastAPI(title="rerank-service", version="1.0.0", lifespan=lifespan)

_model: FlagReranker | None = None


def get_model() -> FlagReranker:
    global _model
    if _model is None:
        model_path = os.getenv("BGE_RERANKER_MODEL_PATH")
        if not model_path:
            raise RuntimeError("环境变量 BGE_RERANKER_MODEL_PATH 未配置")
        device = os.getenv("BGE_RERANKER_DEVICE", "cpu")
        use_fp16 = os.getenv("BGE_RERANKER_FP16", "0") == "1"
        _model = FlagReranker(model_path, device=device, use_fp16=use_fp16)
    return _model


class RerankRequest(BaseModel):
    query: str
    documents: List[str]


class RerankResponse(BaseModel):
    scores: List[float]


@app.post("/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest) -> RerankResponse:
    model = get_model()
    pairs = [[req.query, doc] for doc in req.documents]
    scores = await asyncio.to_thread(model.compute_score, pairs, normalize=True)
    if isinstance(scores, float):
        scores = [scores]
    return RerankResponse(scores=[float(s) for s in scores])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)
