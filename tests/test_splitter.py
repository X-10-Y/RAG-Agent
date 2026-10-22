"""切分逻辑单元测试：表格识别与段落分段"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "backend", "src"))

from rag.splitter import is_table_line, split_into_segments


def test_is_table_line():
    assert is_table_line("| 项目 | 金额 |")
    assert not is_table_line("押金为两个月租金")


def test_split_into_segments():
    segments = split_into_segments(["| 项目 | 金额 |", "押金为两个月租金", ""])
    assert [s[0] for s in segments] == ["table", "text"]
    assert "押金" in segments[1][1]
