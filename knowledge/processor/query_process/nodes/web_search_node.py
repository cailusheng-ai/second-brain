"""联网搜索节点

从 query_rewrite_node 拿到改写后的问题，通过 DashScope MCP 联网搜索服务
（bailian_web_search 工具）获取外部信息，写入 state["web_search_docs"]。

插入位置：multi_search → web_search_node → join
"""

import json
from typing import Any, Dict

import httpx2
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState

load_dotenv()


class WebSearchNode(BaseNode):
    name = "web_search_node"

    async def process(self, state: QueryGraphState) -> QueryGraphState:
        # 联网开关关闭时跳过
        if not state.get("enable_web_search"):
            self.logger.info("联网搜索已关闭，跳过")
            return {"web_search_docs": []}

        # 参数校验，获取前一个节点的 rewritten_query
        rewritten_query = state.get("rewritten_query")
        if not rewritten_query:
            raise StateFieldError(node_name=self.name, field_name="rewritten_query")

        # 异步调用 MCP 工具获取结果，失败降级为空结果
        try:
            mcp_result = await self.call_mcp_execute_web_search(rewritten_query)
        except Exception as e:
            self.logger.warning(f"联网搜索失败，降级为空结果：{e}")
            mcp_result = []

        return {"web_search_docs": mcp_result or []}

    async def call_mcp_execute_web_search(self, rewritten_query: str) -> list[Dict[str, Any]]:
        headers = {"Authorization": f"Bearer {self.config.mcp_dashscope_api_key}"}

        async with httpx2.AsyncClient(headers=headers) as http_client:
            async with streamable_http_client(
                self.config.mcp_dashscope_base_url,
                http_client=http_client,
            ) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tool_result = await session.call_tool(
                        "bailian_web_search",
                        {"query": rewritten_query, "count": 5},
                    )

        content_text = ""
        if tool_result.content:
            content_text = getattr(tool_result.content[0], "text", "") or ""

        data: Dict[str, Any] = json.loads(content_text)

        pages = []
        for page in data.get("pages") or []:
            pages.append({
                "snippet": page.get("snippet"),
                "title": page.get("title"),
                "url": page.get("url"),
            })
        return pages
