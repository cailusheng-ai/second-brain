from langgraph.constants import END
from langgraph.graph import StateGraph

from knowledge.processor.query_process.nodes.dense_search_node import DenseSearchNode
from knowledge.processor.query_process.nodes.sparse_search_node import SparseSearchNode
from knowledge.processor.query_process.nodes.answer_out_put_node import AnswerOutPutNode
from knowledge.processor.query_process.nodes.query_rewrite_node import QueryRewriteNode
from knowledge.processor.query_process.nodes.reranker_node import RerankerNode
from knowledge.processor.query_process.nodes.rrf_node import RRFNode
from knowledge.processor.query_process.nodes.web_search_node import WebSearchNode
from knowledge.processor.query_process.state import QueryGraphState


def create_query_graph():
    """创建查询流程图"""
    workflow = StateGraph(QueryGraphState)

    nodes = {
        "query_rewrite_node": QueryRewriteNode(),
        "multi_search": lambda x: x,
        "dense_search_node": DenseSearchNode(),
        "sparse_search_node": SparseSearchNode(),
        "web_search_node": WebSearchNode(),
        "join": lambda x: {},
        "rrf_node": RRFNode(),
        "reranker_node": RerankerNode(),
        "answer_out_put_node": AnswerOutPutNode(),
    }
    for name, node in nodes.items():
        workflow.add_node(name, node)

    workflow.set_entry_point("query_rewrite_node")

    # 多路搜索分发（dense + sparse + web）
    workflow.add_edge("query_rewrite_node", "multi_search")
    workflow.add_edge("multi_search", "dense_search_node")
    workflow.add_edge("multi_search", "sparse_search_node")
    workflow.add_edge("multi_search", "web_search_node")

    # 多路搜索合并
    workflow.add_edge("dense_search_node", "join")
    workflow.add_edge("sparse_search_node", "join")
    workflow.add_edge("web_search_node", "join")

    # 顺序边
    workflow.add_edge("join", "rrf_node")
    workflow.add_edge("rrf_node", "reranker_node")
    workflow.add_edge("reranker_node", "answer_out_put_node")
    workflow.add_edge("answer_out_put_node", END)

    return workflow.compile()
