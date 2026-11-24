from app.ingestion.embeddings import FakeEmbeddings


def test_deterministic_same_text_same_vector():
    emb = FakeEmbeddings(dim=64)
    v1 = emb.embed(["hello world"])[0]
    v2 = emb.embed(["hello world"])[0]
    assert v1 == v2


def test_vectors_are_normalized():
    emb = FakeEmbeddings(dim=64)
    vec = emb.embed(["some text here"])[0]
    norm = sum(v * v for v in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-6


def test_batch_returns_one_vector_per_text():
    emb = FakeEmbeddings(dim=32)
    out = emb.embed(["a", "b", "c"])
    assert len(out) == 3 and all(len(v) == 32 for v in out)
