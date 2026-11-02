import asyncio
import os.path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.params import Depends
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, StreamingResponse
from starlette.staticfiles import StaticFiles

from knowledge.front.schema.query_schema import QueryRequest, StreamSubmitResponse, QueryResponse
from knowledge.front.schema.login_schema import LoginRequest, LoginResponse
from knowledge.front.service.query_service import QueryService
from knowledge.front.utils.deps import get_query_service
from knowledge.utils.auth import verify_password, create_token, get_current_user
from knowledge.front.utils.paths import get_front_page_dir
from knowledge.processor.query_process.base import setup_logging
from knowledge.utils.sse_util import sse_generator, register_task


def register_router(app):

    #访问前端页面
    @app.get("/chat")
    async def get_chat():
        return FileResponse(os.path.join(get_front_page_dir(),"chat.html"))

    @app.post("/login", response_model=LoginResponse)
    async def login(req: LoginRequest):
        if not verify_password(req.username, req.password):
            raise HTTPException(status_code=401, detail="账号或密码错误")
        return LoginResponse(token=create_token(req.username))

    @app.post("/query", dependencies=[Depends(get_current_user)])
    async def query(request:QueryRequest,
                    service:QueryService=Depends(get_query_service)):
        """处理查询请求"""
        #获取用户问题
        user_query=request.query
        #获取session_id，如果没有就新建一个session_id
        session_id=request.session_id or service.generate_session_id()
        #获取是否流式输出
        is_stream=request.is_stream
        #获取是否启用联网搜索
        enable_web_search=request.enable_web_search

        #生成任务id
        task_id=service.generate_task_id()
        #将该任务id对应的任务的总体状态标记为进行中，创建任务id对应的消息队列
        service.submit_query(task_id)

        #用 asyncio.create_task 启动后台查询协程，并注册到任务表，
        #SSE 检测到前端断开时通过 cancel_task 取消，避免继续烧 token
        task = asyncio.create_task(service.run_query_graph(task_id, session_id, user_query, is_stream, enable_web_search))
        register_task(task_id, task)
        return StreamSubmitResponse(
            message="查询任务已提交，正在查询中",
            session_id=session_id,
            task_id=task_id
        )

    @app.get("/stream/{task_id}")
    async def stream(task_id: str, request: Request):
        return StreamingResponse(
            sse_generator(task_id, request), media_type="text/event-stream",
        )

    @app.get("/history/{session_id}", dependencies=[Depends(get_current_user)])
    async def get_history(
        session_id: str, limit: int = 50,
        service:QueryService=Depends(get_query_service),
    ):
        try:
            items=service.get_history(session_id,limit)
            return {"session_id":session_id,"items":items}
        except Exception as e:
            raise HTTPException(status_code=500,detail=f"history error: {e}")

    @app.delete("/history/{session_id}", dependencies=[Depends(get_current_user)])
    async def clear_chat_history(
            session_id:str,
            service:QueryService=Depends(get_query_service)
    ):
        count=service.clear_history(session_id)
        return {"message": "History cleared", "deleted_count": count}

def create_app()->FastAPI:
    setup_logging()
    #创建app实例
    app=FastAPI(
        title="Query Service",
        description="知识库查询系统"
    )

    # 允许跨域（开发环境）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 允许所有来源
        allow_credentials=True,  # 允许携带凭证
        allow_methods=["*"],  # 允许所有方法
        allow_headers=["*"],  # 允许所有头部
    )

    #静态文件挂载
    page_dir = get_front_page_dir()
    if os.path.exists(page_dir):
        app.mount("/queryfront", StaticFiles(directory=page_dir), name="front")


    #注册路由
    register_router(app)

    return app

app=create_app()

if __name__=="__main__":
    uvicorn.run(app=app,host="0.0.0.0",port=8001)
