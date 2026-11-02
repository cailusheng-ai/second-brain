from langgraph.graph import StateGraph, END

from knowledge.processor.import_process.nodes.chunks_embedding_node import ChunksEmbeddingNode
from knowledge.processor.import_process.nodes.chunks_save_to_milvus_node import ChunksSaveToMilvusNode
from knowledge.processor.import_process.nodes.document_split_node import DocumentSplitNode
from knowledge.processor.import_process.nodes.entry_node import EntryNode
from knowledge.processor.import_process.nodes.md_Img_node import MDImgNode
from knowledge.processor.import_process.nodes.text_clean_node import TextCleanNode
from knowledge.processor.import_process.nodes.topic_tagging_node import TopicTaggingNode
from knowledge.processor.import_process.state import ImportGraphState


def create_import_graph():
    """创建导入流程图（只支持 .md 文件）"""
    workflow = StateGraph(ImportGraphState)

    nodes = {
        "entry_node": EntryNode(),
        "md_Img_node": MDImgNode(),
        "text_clean_node": TextCleanNode(),
        "document_split_node": DocumentSplitNode(),
        "topic_tagging_node": TopicTaggingNode(),
        "chunks_embedding_node": ChunksEmbeddingNode(),
        "chunks_save_to_milvus_node": ChunksSaveToMilvusNode(),
    }

    for name, node in nodes.items():
        workflow.add_node(name, node)

    workflow.set_entry_point("entry_node")

    workflow.add_edge("entry_node", "md_Img_node")
    workflow.add_edge("md_Img_node", "text_clean_node")
    workflow.add_edge("text_clean_node", "document_split_node")
    workflow.add_edge("document_split_node", "topic_tagging_node")
    workflow.add_edge("topic_tagging_node", "chunks_embedding_node")
    workflow.add_edge("chunks_embedding_node", "chunks_save_to_milvus_node")
    workflow.add_edge("chunks_save_to_milvus_node", END)

    return workflow.compile()
