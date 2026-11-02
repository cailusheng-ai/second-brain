from knowledge.processor.query_process.nodes.query_rewrite_node import QueryRewriteNode


def test_parse_result():
    node = QueryRewriteNode()
    result = node._parse_result('{"rewritten_query": "什么是RAG", "topics": ["RAG"]}')
    assert result["rewritten_query"] == "什么是RAG"
    assert result["topics"] == ["RAG"]


def test_parse_result_strips_code_fence():
    node = QueryRewriteNode()
    result = node._parse_result('```json\n{"rewritten_query": "x", "topics": []}\n```')
    assert result["rewritten_query"] == "x"


def test_parse_result_invalid():
    node = QueryRewriteNode()
    assert node._parse_result("不是JSON") == {}


def test_filter_topics():
    node = QueryRewriteNode()
    assert node._filter_topics(["Agent", "不存在"]) == ["Agent"]
    assert node._filter_topics("不是列表") == []
