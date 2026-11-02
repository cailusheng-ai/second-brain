from knowledge.processor.import_process.nodes.chunk_clean_node import ChunkCleanNode


def test_exact_dedup():
    node = ChunkCleanNode()
    chunks = [
        {"body": "内容A", "title": "t1"},
        {"body": "内容A", "title": "t2"},
        {"body": "内容B", "title": "t3"},
    ]
    result = node._exact_dedup(chunks)
    assert len(result) == 2
    assert result[0]["title"] == "t1"
    assert result[1]["title"] == "t3"


def test_assign_quality_flags():
    node = ChunkCleanNode()
    chunks = [
        {"body": "这是一段足够长的正常内容", "title": "ok"},
        {"body": "短", "title": "short"},
        {"body": "", "title": "empty"},
    ]
    result = node._assign_quality_flags(chunks)
    # 空 chunk 默认丢弃
    assert len(result) == 2
    assert result[0]["clean_quality_flag"] == "ok"
    assert result[1]["clean_quality_flag"] == "too_short"
