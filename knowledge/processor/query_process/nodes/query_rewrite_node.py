"""问题改写与主题识别节点

查询流程入口：结合历史对话改写用户问题（消解代词），并从主题分类表
识别用户问题涉及的主题标签，写入 state。

插入位置：query_rewrite_node → multi_search → {dense, sparse, web}
"""

import asyncio
import json
import logging
import re

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompts.query.query_prompt import (
    QUERY_REWRITE_SYSTEM_PROMPT,
    QUERY_REWRITE_TEMPLATE,
)
from knowledge.utils.llm_client_utils import get_llm_client
from knowledge.utils.mongodb_client_utils import get_recent_message, save_chat_message

load_dotenv()
logger = logging.getLogger(__name__)


class QueryRewriteNode(BaseNode):
    """结合历史对话改写问题 + 识别主题标签。"""

    name = "query_rewrite_node"

    async def process(self, state: QueryGraphState) -> QueryGraphState:
        original_query = state.get("original_query", "")
        session_id = state.get("session_id", "")

        # 1. 获取最近对话历史
        chat_history = await asyncio.to_thread(get_recent_message, session_id)

        # 2. LLM 改写问题 + 识别主题标签
        rewritten_query, topics = await self._rewrite_and_tag(original_query, chat_history)

        # 3. 保存用户问题（含改写结果与主题标签）
        await asyncio.to_thread(
            save_chat_message,
            session_id,
            "user",
            original_query,
            rewritten_query=rewritten_query,
            topics=topics,
        )

        state["history"] = chat_history
        state["rewritten_query"] = rewritten_query
        state["topics"] = topics
        return state

    async def _rewrite_and_tag(self, original_query: str, chat_history: list):
        history_text = ""
        for message in chat_history:
            role = message.get("role")
            text = message.get("text")
            history_text += f"{role}:{text}\n"

        taxonomy = self.config.topic_taxonomy or []
        user_prompt = QUERY_REWRITE_TEMPLATE.format(
            history_text=history_text,
            query=original_query,
            taxonomy="、".join(taxonomy),
        )
        messages = [
            SystemMessage(content=QUERY_REWRITE_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            llm_client = get_llm_client(response_format=True)
            response = await llm_client.ainvoke(messages)
            result = self._parse_result(getattr(response, "content", ""))
        except Exception as e:
            logger.error(f"问题改写失败，回退到原始问题：{e}")
            return original_query, []

        rewritten_query = result.get("rewritten_query") or original_query
        topics = self._filter_topics(result.get("topics"))
        return rewritten_query, topics

    def _parse_result(self, content: str) -> dict:
        content = (content or "").strip()
        content = re.sub(r"^```[\s\w]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content).strip()
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                return {}
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return {}
        return data if isinstance(data, dict) else {}

    def _filter_topics(self, raw) -> list:
        if not isinstance(raw, list):
            return []
        allowed = set(self.config.topic_taxonomy or [])
        topics = [str(t).strip() for t in raw if str(t).strip() in allowed]
        return topics[:3]
