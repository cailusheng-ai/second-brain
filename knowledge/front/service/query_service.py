import logging
import uuid
from typing import List, Dict, Any

from knowledge.front.service.task_service import TaskService
from knowledge.front.utils.task_util import TASK_STATUS_PROCESSING, update_task_status
from knowledge.processor.query_process.main_graph import create_query_graph
from knowledge.utils.mongodb_client_utils import get_recent_message, delete_chat_message
from knowledge.utils.sse_util import get_or_create_sse_queue, push_sse_message, SSEEvent

# 节点名称中文映射
NODE_NAME_CN = {
    "query_rewrite_node": "问题改写与主题识别",
    "multi_search": "多路检索分发",
    "dense_search_node": "稠密检索（语义）",
    "sparse_search_node": "稀疏检索（关键词）",
    "web_search_node": "联网搜索",
    "join": "多路结果汇聚",
    "rrf_node": "粗排融合（RRF）",
    "reranker_node": "精排排序（Reranker）",
    "answer_out_put_node": "答案生成",
}

# 节点后继关系（用于 running_list：某节点完成后，标记其后继节点为"进行中"）
NEXT_NODES = {
    "query_rewrite_node": ["multi_search"],
    "multi_search": ["dense_search_node", "sparse_search_node", "web_search_node"],
    "dense_search_node": ["join"],
    "sparse_search_node": ["join"],
    "web_search_node": ["join"],
    "join": ["rrf_node"],
    "rrf_node": ["reranker_node"],
    "reranker_node": ["answer_out_put_node"],
    "answer_out_put_node": [],
}


class QueryService:
    logger=logging.getLogger(__name__)
    def __init__(self,task_service:TaskService):
        self.task_service=task_service


    #生成session_id和task_id
    def generate_session_id(self)->str:
        return str(uuid.uuid4())

    def generate_task_id(self) -> str:
        return str(uuid.uuid4())


    def submit_query(self,task_id:str):

        #更新任务状态为任务处理中
        update_task_status(task_id,TASK_STATUS_PROCESSING)

        # 总是创建消息队列：节点进度 + 答案都通过它推给 SSE（无论流不流式）
        get_or_create_sse_queue(task_id)

    async def run_query_graph(self,task_id,session_id,query,is_stream,enable_web_search=False):
        #构建初始状态
        initial_state={
            "task_id":task_id,
            "session_id":session_id,
            "original_query":query,
            "is_stream":is_stream,
            "enable_web_search":enable_web_search
        }
        #构建图实例
        query_agent=create_query_graph()

        # stream_mode="updates"：每个 event 是 {节点名: 节点状态更新}；
        # 默认 "values" 是完整状态（key 是状态字段名），下面按节点名取会错位
        done_eng = []
        running_eng = []

        try:
            async for event in query_agent.astream(initial_state, stream_mode="updates"):
                for name, node_state in event.items():
                    # 当前节点完成：移出 running、加入 done
                    if name in running_eng:
                        running_eng.remove(name)
                    if name not in done_eng:
                        done_eng.append(name)
                    self.task_service.mark_node_done(task_id, name)
                    self.logger.info(f"{task_id}完成了节点{NODE_NAME_CN.get(name, name)}")

                    # 标记后继节点为"进行中"（并行扇出会一次标多个）
                    next_nodes = NEXT_NODES.get(name, [])
                    for next_name in next_nodes:
                        if next_name not in done_eng and next_name not in running_eng:
                            running_eng.append(next_name)
                        self.task_service.mark_node_running(task_id, next_name)

                    # 推送进度给前端（DONE 事件触发后队列可能已被清理，加保护）
                    try:
                        push_sse_message(task_id, SSEEvent.PROGRESS, {
                            "done_list": [NODE_NAME_CN.get(n, n) for n in done_eng],
                            "running_list": [NODE_NAME_CN.get(n, n) for n in running_eng],
                            "status": "processing"
                        })
                    except ValueError:
                        pass

            # 正常完成：推送最终完成状态
            try:
                push_sse_message(task_id, SSEEvent.PROGRESS, {
                    "done_list": [NODE_NAME_CN.get(n, n) for n in done_eng],
                    "running_list": [],
                    "status": "completed"
                })
            except ValueError:
                pass

        except Exception as e:
            error_msg = str(e)
            self.logger.exception(f"任务 {task_id} 执行失败: {error_msg}")
            try:
                push_sse_message(task_id, SSEEvent.PROGRESS, {
                    "done_list": [NODE_NAME_CN.get(n, n) for n in done_eng],
                    "running_list": [],
                    "status": "failed"
                })
            except ValueError:
                pass
            try:
                push_sse_message(task_id, SSEEvent.MESSAGE, {"answer": "", "error": error_msg})
            except ValueError:
                pass
        finally:
            try:
                from knowledge.utils.sse_util import signal_done
                signal_done(task_id, {"answer": "", "error": ""})
            except ValueError:
                pass

    def get_history(self, session_id: str, limit: int = 50
                      ) -> List[Dict[str, Any]]:
        records = get_recent_message(session_id, limit=limit)
        return [
            {
                "_id": str(r.get("_id", "")),
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "topics": r.get("topics", []),
                "sources": r.get("sources", []),
                "ts": r.get("ts"),
            }
            for r in records
        ]

    def clear_history(self, session_id: str) -> int:
        return delete_chat_message(session_id)
