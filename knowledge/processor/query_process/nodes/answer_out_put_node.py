import asyncio
import logging
import re
from typing import List, Dict, Any

from knowledge.processor.query_process.base import BaseNode, T
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query.query_prompt import ANSWER_PROMPT
from knowledge.utils.llm_client_utils import get_llm_client
from knowledge.utils.mongodb_client_utils import save_chat_message
from knowledge.utils.sse_util import push_sse_message, SSEEvent

logger = logging.getLogger(__name__)


class AnswerOutPutNode(BaseNode):
    name = "answer_out_put_node"


    async def process(self, state: QueryGraphState) -> QueryGraphState:
        task_id=state.get("task_id")

        sources = []
        #1、判断是否已有答案（捷径/候选反问）
        if state.get("answer"):
            push_sse_message(task_id,SSEEvent.MESSAGE,{"answer":state.get("answer")})
        else:
            #2、确定最终检索文档（截断后），构建提示词与引用来源
            docs = self._truncate_docs(state.get("reranked_docs") or state.get("rrf_chunks") or [])
            prompt=self._build_prompt(state, docs)
            state["prompt"]=prompt
            all_sources=self._build_sources(docs)
            #3、调用大模型、生成答案（流式逐token / 非流式整段，均走 SSE）
            await self._call_llm_generate_answer(prompt,state)
            #3.5、只返回答案里实际标注引用的来源，而非全部检索到的 chunk
            sources=self._filter_cited_sources(all_sources, state.get("answer", ""))

        #8、保存历史对话
        if state.get("answer"):
            state["sources"] = sources
            await self._write_history(state)

        #9、统一推送结束事件（无论流不流式），携带完整答案与引用来源
        push_sse_message(task_id,SSEEvent.DONE, {"answer":state.get("answer"),"sources":sources,"status":"completed"})

        return state



    def _build_prompt(self, state: QueryGraphState, docs) -> str:
        """构建 LLM 提示词。

        从 state 中提取排序文档、主题标签、历史对话、用户问题，
        按 ANSWER_PROMPT 模板组装为完整的提示词字符串。

        组装逻辑：
          - context:    排序文档 → 可读文本（含排名、标题、内容）
          - history:    历史对话 → "user: ...\\nassistant: ..."
          - topics: 主题标签列表 → "主题A、主题B"
          - question:   rewritten_query 优先 → original_query 降级

        Args:
            state: 查询图状态。

        Returns:
            完整的提示词字符串，可直接传入 LLM。
        """
        # ---- 1. 获取排序文档，格式化为上下文字符串 ----
        context = self._build_context(docs)

        # ---- 2. 格式化历史对话 ----
        history_text = self._build_history(state)

        # ---- 3. 格式化主题标签 ----
        topics = state.get("topics") or []
        topics_text = "、".join(topics) if topics else "未指定"

        # ---- 4. 获取用户问题（改写后优先） ----
        question = (
            state.get("rewritten_query")
            or state.get("original_query")
            or ""
        )

        # ---- 5. 填充模板 ----
        prompt = ANSWER_PROMPT.format(
            context=context,
            history=history_text,
            topics=topics_text,
            question=question,
        )

        logger.info(
            f"提示词构建完成: question={question[:60]}..., "
            f"context={len(context)} 字符, "
            f"history={len(history_text.split(chr(10))) - 1} 条"
        )

        return prompt

    # ========================================================================
    # _build_prompt 的子步骤
    # ========================================================================

    def _truncate_docs(self, docs):
        """按 max_context_chars 截断文档列表，供 context 与 sources 共用，保证引用编号一致。"""
        max_chars = self.config.max_context_chars
        result = []
        total = 0
        for doc in docs:
            result.append(doc)
            title = doc.get("title", "")
            body = doc.get("content") or doc.get("snippet") or ""
            total += len(title) + len(body)
            if max_chars > 0 and total >= max_chars:
                logger.info(f"上下文截断: 保留 {len(result)}/{len(docs)} 条")
                break
        return result

    def _build_context(self, docs) -> str:
        """将排序文档格式化为 LLM 可读的上下文字符串（带引用编号）。

        每条文档前标注 [i] 编号，供 LLM 在答案中引用。
        无文档时返回 "暂无相关参考内容。"。
        """
        if not docs:
            return "暂无相关参考内容。"

        blocks: List[str] = []
        for i, doc in enumerate(docs, start=1):
            # 排名：精排 > 粗排 > 未知
            rank = (
                doc.get("rerank_final_rank")
                or doc.get("rrf_final_rank")
                or "?"
            )
            # 分数
            score = doc.get("rerank_score") or doc.get("rrf_score") or 0
            # 标题
            title = doc.get("title", "")
            title = " ".join(title.split()) if title else "无标题"
            # 正文：优先 content（chunk 结构），其次 snippet（web 结果）
            body = doc.get("content") or doc.get("snippet") or ""
            body = " ".join(body.split()) if body else "无内容"

            block = (
                f"[{i}] 文档 #{rank}（相关度: {score:.4f}）\n"
                f"标题: {title}\n"
                f"内容: {body}"
            )
            blocks.append(block)

        return "\n\n".join(blocks)

    def _build_sources(self, docs) -> List[Dict[str, Any]]:
        """根据截断后的文档构建引用来源列表（区分文档标题与 chunk 章节标题）。"""
        sources = []
        for i, doc in enumerate(docs, start=1):
            # 文档标题：内部 chunk 用 file_title，web 结果用网页 title
            doc_title = doc.get("file_title") or doc.get("title") or ""
            doc_title = " ".join(doc_title.split()) if doc_title else "无标题"
            # chunk 章节标题：仅内部 chunk 有；web 结果无 chunk 概念
            chunk_title = doc.get("title", "") if doc.get("file_title") else ""
            chunk_title = " ".join(chunk_title.split()) if chunk_title else ""
            body = doc.get("content") or doc.get("snippet") or ""
            body = " ".join(body.split()) if body else ""
            sources.append({
                "index": i,
                "doc_title": doc_title,
                "chunk_title": chunk_title,
                "chapter_path": doc.get("chapter_path", ""),
                "url": doc.get("url", ""),
                "chunk_id": str(doc.get("chunk_id")) if doc.get("chunk_id") is not None else None,
                "snippet": body,
                "score": round(doc.get("rerank_score") or doc.get("rrf_score") or 0, 4),
            })
        return sources

    def _filter_cited_sources(self, all_sources, answer):
        """返回答案实际引用的来源；若答案未标注引用编号，则返回全部来源（兜底）。"""
        if not answer:
            return all_sources
        cited = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
        if not cited:
            return all_sources
        return [s for s in all_sources if s["index"] in cited]

    def _build_history(self, state: QueryGraphState) -> str:
        """将历史对话列表格式化为文本。

        格式：每行一条 "role: text"，最多取最近 10 条，
        单条超过 200 字符自动截断。

        Args:
            state: 查询图状态。

        Returns:
            格式化的历史对话文本，无历史时返回 "无历史对话"。
        """
        history: List[Dict[str, Any]] = state.get("history") or []

        if not history:
            return "无历史对话"

        lines = []
        for msg in history[-10:]:
            role = msg.get("role", "unknown")
            text = msg.get("text", "")
            if len(text) > 200:
                text = text[:200] + "..."
            lines.append(f"{role}: {text}")

        return "\n".join(lines)



    async def _call_llm_generate_answer(self, prompt, state):
        llm_client=get_llm_client()
        task_id = state.get("task_id")
        is_stream=state.get("is_stream")

        if is_stream:
            # 流式：逐 token 推 MESSAGE，前端打字机展示
            stream_result=await self.stream_generate_result(llm_client,prompt,task_id)
            state["answer"]=stream_result
        else:
            # 非流式：整段生成后作为一条 MESSAGE 推送
            invoke_result=await self.invoke_generate_result(llm_client,prompt)
            state["answer"] = invoke_result
            push_sse_message(task_id, SSEEvent.MESSAGE, {"answer": invoke_result})

    async def stream_generate_result(self, llm_client, prompt, task_id):
        stream_result=""
        try:
            async for chunk in llm_client.astream(prompt):
                text=getattr(chunk,"content")
                if text:
                    stream_result +=text
                    push_sse_message(task_id,SSEEvent.MESSAGE,{"message":text})
        except Exception as e:
            self.logger.error(f"流式生成错误：{e}")

        return stream_result

    async def invoke_generate_result(self, llm_client, prompt):
        response=await llm_client.ainvoke(prompt)
        invoke_result=response.content
        return invoke_result

    async def _write_history(self, state: QueryGraphState) -> None:
        """将 assistant 回答持久化到 MongoDB。

        用户问题已在 query_rewrite_node 中保存，这里只保存回答。
        """
        session_id = state.get("session_id", "")
        if not session_id:
            logger.warning("session_id 为空，跳过对话保存")
            return

        rewritten_query = state.get("rewritten_query", "")
        topics = state.get("topics") or []
        answer = state.get("answer", "")
        sources = state.get("sources") or []

        try:
            await asyncio.to_thread(
                save_chat_message,
                session_id=session_id,
                role="assistant",
                text=answer,
                rewritten_query=rewritten_query,
                topics=topics,
                sources=sources,
            )
            logger.info(f"AI 回答已保存: session={session_id}")
        except Exception as e:
            # 保存失败不阻断主流程（答案已经推送给用户了）
            logger.exception(f"对话保存失败（不影响答案输出）: {e}")
