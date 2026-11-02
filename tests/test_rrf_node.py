from knowledge.processor.query_process.nodes.rrf_node import RRFNode


def test_get_doc_id_chunk():
    node = RRFNode()
    assert node._get_doc_id({"chunk_id": 5}) == "chunk_5"


def test_get_doc_id_web():
    node = RRFNode()
    assert node._get_doc_id({"url": "https://a.com"}) == "web_https://a.com"


def test_get_doc_id_hash():
    node = RRFNode()
    assert node._get_doc_id({"snippet": "hello"}).startswith("hash_")


def test_collect_and_dedup():
    node = RRFNode()
    sources = {
        "dense": [{"chunk_id": 1, "content": "A"}],
        "sparse": [{"chunk_id": 1, "content": "A"}],  # 与 dense 命中同一 chunk
        "web": [{"url": "https://a.com", "snippet": "A"}],
    }
    doc_map, rank_map = node._collect_and_dedup(sources)
    # dense/sparse 命中同一 chunk → 去重为 1 条，web 另算 1 条
    assert len(doc_map) == 2
    assert rank_map["chunk_1"]["dense"] == 1
    assert rank_map["chunk_1"]["sparse"] == 1
    assert rank_map["web_https://a.com"]["web"] == 1
    # 多路命中的 chunk 应标记两个来源
    assert set(doc_map["chunk_1"]["_rrf_sources"]) == {"dense", "sparse"}


def test_compute_rrf_scores():
    node = RRFNode()
    rank_map = {"d1": {"dense": 1}, "d2": {"dense": 2}}
    weights = {"dense": 1.0}
    k_values = {"dense": 60}
    scores = node._compute_rrf_scores(rank_map, weights, k_values)
    assert scores["d1"] > scores["d2"]


def test_compute_rrf_scores_multi_source():
    """同一文档被多路命中时，RRF 分数应累加各路贡献。"""
    node = RRFNode()
    # d1 同时被 dense 和 sparse 命中，d2 只被 dense 命中
    rank_map = {"d1": {"dense": 1, "sparse": 2}, "d2": {"dense": 1}}
    weights = {"dense": 1.0, "sparse": 1.0}
    k_values = {"dense": 60, "sparse": 60}
    scores = node._compute_rrf_scores(rank_map, weights, k_values)
    assert scores["d1"] > scores["d2"]
