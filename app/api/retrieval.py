"""检索路由：混合检索接口与带引用问答接口。"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.deps import get_current_user
from app.models import RetrievalLog, User
from app.retrieval.generator import generate_answer
from app.retrieval.hybrid import retrieve
from app.retrieval.reranker import get_reranker
from app.schemas import (
    AskRequest,
    AskResponse,
    ScoredChunk,
    SearchRequest,
    SearchResponse,
)

router = APIRouter(tags=["retrieval"])


@router.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> SearchResponse:
    candidates = retrieve(db, payload.query, strategy=payload.strategy)
    top = payload.top_k or settings.final_top_k
    results = [
        ScoredChunk(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            document_title=c.document_title,
            page=c.page,
            score=round(c.rrf_score, 6),
            content=c.content,
        )
        for c in candidates[:top]
    ]
    return SearchResponse(query=payload.query, strategy=payload.strategy, results=results)


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> AskResponse:
    # 1) 混合召回（向量 + 关键词，RRF 融合）
    candidates = retrieve(db, payload.query, strategy="hybrid")
    # 2) 精排：重排模型在候选集上再打分
    reranked = get_reranker().rerank(payload.query, candidates, top_n=settings.rerank_top_n)
    # 3) 证据门控 + 生成回答（证据不足拒答）
    answer, refused, citations = generate_answer(payload.query, reranked)

    db.add(
        RetrievalLog(
            query=payload.query,
            user_id=user.id,
            strategy="hybrid",
            refused=refused,
            cited_chunk_ids=[c["chunk_id"] for c in citations],
            answer=answer,
        )
    )
    db.commit()

    # 补充来源文件名
    from app.models import Document

    names = {d.id: d.source_filename for d in db.query(Document.id, Document.source_filename).all()}
    for cite in citations:
        cite["source_filename"] = names.get(cite["document_id"])

    return AskResponse(
        query=payload.query,
        answer=answer,
        refused=refused,
        citations=citations,
        used_strategy="hybrid",
    )
