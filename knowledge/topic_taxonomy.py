"""主题分类表 — 入库打标与查询识别共用的单一来源。

如需增删主题，只需修改本文件；入库（topic_tagging_node）与查询
（query_rewrite_node）会自动跟随。
"""

TOPIC_TAXONOMY = [
    "LLM 基础", "Agent", "RAG", "Prompt 工程", "微调", "工具调用", "推理", "评测", "Java基础", "Java框架", "其他",
]
