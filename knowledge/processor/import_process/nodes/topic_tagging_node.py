"""主题标签节点

在文档切分之后，取前 K 段内容交给 LLM，从主题分类表里为文档
挑选 1-3 个主题标签，写入 state["topics"] 和每个 chunk 的 "topics" 字段。

插入位置：document_split_node → topic_tagging_node → chunks_embedding_node
"""

import json
import logging
import re

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.prompts.upload.import_prompt import (
    TOPIC_TAG_SYSTEM_PROMPT,
    TOPIC_TAG_USER_TEMPLATE,
)
from knowledge.utils.llm_client_utils import get_llm_client

load_dotenv()
logger = logging.getLogger(__name__)


class TopicTaggingNode(BaseNode):
    name = "topic_tagging_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 校验输入
        file_title, chunks = self._validate_input(state)

        # 2. 取前 K 段内容作为打标依据
        context = self._get_part_chunks(chunks)

        # 3. 调用 LLM 从分类表挑选主题标签
        topics = self._tag_topics_by_llm(file_title, context)

        # 4. 写入 state 和每个 chunk
        self._fill_topics(topics, state, chunks)
        return state

    def _validate_input(self, state: ImportGraphState):
        chunks = state.get("chunks")
        file_title = state.get("file_title")
        if not chunks or not isinstance(chunks, list):
            raise ValidationError("chunks 为空或不是列表", self.name)
        if not file_title:
            raise ValidationError("文件标题为空", self.name)
        if self.config.topic_chunk_k <= 0:
            raise ValidationError("topic_chunk_k 为空或无效", self.name)
        self.logger.info(f"检测到文件 {file_title}，对应切片长度 {len(chunks)}")
        return file_title, chunks

    def _get_part_chunks(self, chunks: list) -> str:
        total = 0
        result = []
        for index, chunk in enumerate(chunks[: self.config.topic_chunk_k]):
            if not isinstance(chunk, dict) or not chunk:
                continue
            content = chunk.get("body") or chunk.get("content") or ""
            if not content:
                continue
            piece = f"【切片】{index + 1}-{content}"
            result.append(piece)
            total += len(piece)
            if total > self.config.topic_chunk_size:
                break
        return "\n\n".join(result)[: self.config.topic_chunk_size]

    def _tag_topics_by_llm(self, file_title: str, context: str) -> list:
        taxonomy = self.config.topic_taxonomy or []
        user_prompt = TOPIC_TAG_USER_TEMPLATE.format(
            taxonomy="、".join(taxonomy),
            file_title=file_title,
            context=context,
        )
        messages = [
            SystemMessage(content=TOPIC_TAG_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        try:
            llm_client = get_llm_client(response_format=True)
            response = llm_client.invoke(messages)
            topics = self._parse_topics(getattr(response, "content", ""))
        except Exception as e:
            logger.error(f"主题标签识别失败，回退到「其他」：{e}")
            return ["其他"]

        if not topics:
            logger.info("未能识别有效主题标签，回退到「其他」")
            return ["其他"]

        self.logger.info(f"识别到主题标签：{topics}")
        return topics

    def _parse_topics(self, content: str) -> list:
        content = (content or "").strip()
        content = re.sub(r"^```[\s\w]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content).strip()
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                return []
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return []

        raw = data.get("topics") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            return []

        allowed = set(self.config.topic_taxonomy or [])
        topics = [str(t).strip() for t in raw if str(t).strip() in allowed]
        return topics[: self.config.topic_max_tags]

    def _fill_topics(self, topics: list, state: ImportGraphState, chunks: list):
        state["topics"] = topics
        topics_str = ",".join(topics)
        for chunk in chunks:
            chunk["topics"] = topics_str
