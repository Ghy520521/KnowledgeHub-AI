"""混合检索：向量召回 + 关键词召回，RRF 融合。

- PostgreSQL 路径：pgvector 余弦距离 + to_tsvector/plainto_tsquery 全文检索。
- SQLite 回退路径（本地/测试）：Python 内计算余弦与词面重叠，保证逻辑可跑通。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import settings
from app.ingestion.embeddings import get_embedder
from app.models import Chunk, Document
from app.retrieval.rrf import rrf_fuse


@dataclass
class Candidate:
    chunk_id: int
    document_id: int
    document_title: str
    page: int | None
    content: str
    rrf_score: float = 0.0
    vector_rank: int | None = None
    keyword_rank: int | None = None
    rerank_score: float | None = None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5 or 1.0
    nb = sum(y * y for y in b) ** 0.5 or 1.0
    return dot / (na * nb)


def _vector_search(db: Session, query_vec: list[float], top_k: int) -> list[tuple[int, float]]:
    if settings.is_postgres:
        dist = Chunk.embedding.cosine_distance(query_vec)
        rows = (
            db.query(Chunk.id, dist.label("d"))
            .order_by(dist)
            .limit(top_k)
            .all()
        )
        return [(cid, 1.0 - float(d)) for cid, d in rows]

    scored: list[tuple[int, float]] = []
    for chunk in db.query(Chunk).all():
        if chunk.embedding:
            scored.append((chunk.id, _cosine(query_vec, chunk.embedding)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def _keyword_search(db: Session, query: str, top_k: int) -> list[tuple[int, float]]:
    if settings.is_postgres:
        rank = func.ts_rank(func.to_tsvector("simple", Chunk.content), func.plainto_tsquery("simple", query))
        rows = (
            db.query(Chunk.id, rank.label("r"))
            .order_by(rank.desc())
            .limit(top_k)
            .all()
        )
        return [(cid, float(r)) for cid, r in rows if r > 0]

    tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", query.lower()))
    scored: list[tuple[int, float]] = []
    for chunk in db.query(Chunk).all():
        body = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", chunk.content.lower()))
        overlap = len(tokens & body)
        if overlap:
            scored.append((chunk.id, overlap / (len(tokens) or 1)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def retrieve(
    db: Session,
    query: str,
    strategy: str = "hybrid",
    vector_top_k: int | None = None,
    fts_top_k: int | None = None,
) -> list[Candidate]:
    """执行召回并返回融合后的候选（未重排）。"""
    vector_top_k = vector_top_k or settings.vector_top_k
    fts_top_k = fts_top_k or settings.fts_top_k

    embedder = get_embedder()
    query_vec = embedder.embed([query])[0]

    vector_hits = _vector_search(db, query_vec, vector_top_k) if strategy in ("vector", "hybrid") else []
    keyword_hits = _keyword_search(db, query, fts_top_k) if strategy in ("keyword", "hybrid") else []

    # 记录各检索器排名，供 RRF 与调试使用
    rank_map: dict[int, dict[str, int]] = {}
    for rank, (cid, _) in enumerate(vector_hits, start=1):
        rank_map.setdefault(cid, {})["vector"] = rank
    for rank, (cid, _) in enumerate(keyword_hits, start=1):
        rank_map.setdefault(cid, {})["keyword"] = rank

    if strategy == "hybrid":
        fused = rrf_fuse([vector_hits, keyword_hits], k=settings.rrf_k)
    elif strategy == "vector":
        fused = [(cid, score) for cid, score in vector_hits]
    else:
        fused = [(cid, score) for cid, score in keyword_hits]

    candidates: list[Candidate] = []
    chunk_ids = [cid for cid, _ in fused]
    chunks = db.query(Chunk).filter(Chunk.id.in_(chunk_ids)).all()
    chunk_by_id = {c.id: c for c in chunks}
    doc_titles = {
        d.id: d.title for d in db.query(Document.id, Document.title).all()
    }

    for cid, score in fused:
        chunk = chunk_by_id.get(cid)
        if chunk is None:
            continue
        candidates.append(
            Candidate(
                chunk_id=cid,
                document_id=chunk.document_id,
                document_title=doc_titles.get(chunk.document_id, ""),
                page=chunk.page,
                content=chunk.content,
                rrf_score=score,
                vector_rank=rank_map.get(cid, {}).get("vector"),
                keyword_rank=rank_map.get(cid, {}).get("keyword"),
            )
        )
    return candidates
