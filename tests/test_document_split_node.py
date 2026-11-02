from knowledge.processor.import_process.nodes.document_split_node import DocumentSplitNode


def test_is_toc_entry():
    node = DocumentSplitNode()
    assert node._is_toc_entry("第一章 产品概述 1")
    assert node._is_toc_entry("1.1 产品简介 3")
    assert not node._is_toc_entry("这是一段正文内容")
    assert not node._is_toc_entry("# 标题")


def test_build_chapter_path():
    levels = ["", "# 第一章", "## 1.1", "", "", "", ""]
    assert DocumentSplitNode._build_chapter_path(levels, 2) == "第一章 > 1.1"
