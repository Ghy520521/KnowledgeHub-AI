from app.ingestion.chunker import chunk_text


def test_chunks_within_size():
    text = "句子一。" * 300
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert chunks
    for c in chunks:
        assert len(c) <= 100


def test_overlap_between_adjacent_chunks():
    text = "甲" * 300
    chunks = chunk_text(text, chunk_size=80, chunk_overlap=20)
    assert len(chunks) >= 2
    # 相邻块尾部与头部存在重叠字符
    assert chunks[0][-10:] in chunks[1]


def test_empty_text_returns_empty():
    assert chunk_text("") == []


def test_overlap_must_be_smaller_than_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=10, chunk_overlap=10)
