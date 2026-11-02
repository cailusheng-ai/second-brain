from knowledge.processor.import_process.nodes.topic_tagging_node import TopicTaggingNode


def test_parse_topics_valid():
    node = TopicTaggingNode()
    assert node._parse_topics('{"topics": ["Agent", "RAG"]}') == ["Agent", "RAG"]


def test_parse_topics_filters_unknown():
    node = TopicTaggingNode()
    assert node._parse_topics('{"topics": ["Agent", "不存在主题", "其他"]}') == ["Agent", "其他"]


def test_parse_topics_truncates_to_max():
    node = TopicTaggingNode()
    assert node._parse_topics('{"topics": ["Agent", "RAG", "微调", "多模态"]}') == ["Agent", "RAG", "微调"]


def test_parse_topics_invalid_json():
    node = TopicTaggingNode()
    assert node._parse_topics("不是JSON") == []
