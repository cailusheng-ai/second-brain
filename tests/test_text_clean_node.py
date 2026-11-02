from knowledge.processor.import_process.nodes.text_clean_node import TextCleanNode


def test_trad_to_simp():
    node = TextCleanNode()
    assert node._trad_to_simp("萬用表") == "万用表"


def test_fullwidth_to_halfwidth():
    node = TextCleanNode()
    assert node._fullwidth_to_halfwidth("ＡＢＣ１２３") == "ABC123"


def test_remove_control_chars():
    node = TextCleanNode()
    text, removed = node._remove_control_chars("a\x00b\x01c")
    assert "\x00" not in text and "\x01" not in text
    assert removed == 2


def test_remove_zero_width_chars():
    node = TextCleanNode()
    text, removed = node._remove_zero_width_chars("a​b")
    assert "​" not in text
    assert removed == 1


def test_merge_blank_lines():
    node = TextCleanNode()
    text, _ = node._merge_blank_lines("a\n\n\n\n\nb", max_blank=1)
    assert text == "a\n\nb"
