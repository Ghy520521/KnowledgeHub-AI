"""端到端（SQLite + fake 适配器）：摄入 -> 混合召回 -> 精排 -> 带引用问答/拒答。"""
from app.ingestion.pipeline import process_document, register_pending_bytes
from app.models import Document
from app.retrieval.generator import generate_answer
from app.retrieval.hybrid import retrieve
from app.retrieval.reranker import get_reranker


def _ingest(db, filename: str, content: str) -> int:
    doc = Document(title=filename, source_filename=filename, file_type="txt", status="pending")
    db.add(doc)
    db.commit()
    db.refresh(doc)
    register_pending_bytes(doc.id, content.encode("utf-8"))
    process_document(db, doc.id)
    return doc.id


def test_hybrid_search_finds_relevant_doc(db_session):
    _ingest(db_session, "buds_faq.txt", "耳机左右断连：放回充电盒 10 秒重新取出，重新配对并升级固件。")
    _ingest(db_session, "policy.md", "7 天无理由退货，定制商品除外，退款 3-5 个工作日到账。")

    candidates = retrieve(db_session, "耳机断连怎么处理", strategy="hybrid")
    assert candidates
    top = candidates[0]
    assert "断连" in top.content or "耳机" in top.content


def test_answer_with_citations_when_evidence_sufficient(db_session):
    _ingest(db_session, "buds_faq.txt", "耳机断连：放回充电盒 10 秒后重新取出并重新配对。")
    candidates = retrieve(db_session, "耳机断连怎么办", strategy="hybrid")
    reranked = get_reranker().rerank("耳机断连怎么办", candidates, top_n=5)
    answer, refused, citations = generate_answer("耳机断连怎么办", reranked)
    assert refused is False
    assert citations
    assert citations[0]["chunk_id"] == reranked[0].chunk_id
    assert answer


def test_refuse_when_evidence_insufficient(db_session):
    _ingest(db_session, "buds_faq.txt", "耳机断连：放回充电盒重新配对。")
    candidates = retrieve(db_session, "如何在火星上种植水稻", strategy="hybrid")
    reranked = get_reranker().rerank("如何在火星上种植水稻", candidates, top_n=5)
    answer, refused, citations = generate_answer("如何在火星上种植水稻", reranked)
    assert refused is True
    assert citations == []
    assert "证据不足" in answer
