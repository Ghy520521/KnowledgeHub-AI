"""离线检索评测：Recall@K、MRR、nDCG@K，对比不同检索策略。"""
from __future__ import annotations

import math
import time

from sqlalchemy.orm import Session

from app.config import settings
from app.retrieval.hybrid import Candidate, retrieve
from app.retrieval.reranker import get_reranker


# ---------- 指标 ----------
def recall_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    hit = sum(1 for cid in ranked[:k] if cid in relevant)
    return hit / len(relevant) if relevant else 0.0


def mrr(ranked: list[int], relevant: set[int]) -> float:
    for rank, cid in enumerate(ranked, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    dcg = 0.0
    for rank, cid in enumerate(ranked[:k], start=1):
        if cid in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal > 0 else 0.0


# ---------- 评测执行 ----------
def _ranked_ids(candidates: list[Candidate]) -> list[int]:
    return [c.chunk_id for c in candidates]


def run_eval(
    db: Session,
    samples: list[dict],
    strategies: tuple[str, ...] = ("keyword", "vector", "hybrid", "hybrid_rerank"),
    k: int = settings.final_top_k,
) -> dict:
    """samples: [{"query": str, "relevant_chunk_ids": [int, ...]}, ...]"""
    report: dict = {
        "k": k,
        "sample_count": len(samples),
        "strategies": {},
    }

    for strategy in strategies:
        per_query = {"recall@k": [], "mrr": [], "ndcg@k": [], "latency_ms": []}
        for sample in samples:
            t0 = time.perf_counter()
            if strategy == "hybrid_rerank":
                cands = retrieve(db, sample["query"], strategy="hybrid")
                cands = get_reranker().rerank(sample["query"], cands, top_n=settings.rerank_top_n)
            else:
                cands = retrieve(db, sample["query"], strategy=strategy)
            per_query["latency_ms"].append((time.perf_counter() - t0) * 1000)

            ranked = _ranked_ids(cands)
            relevant = set(sample["relevant_chunk_ids"])
            per_query["recall@k"].append(recall_at_k(ranked, relevant, k))
            per_query["mrr"].append(mrr(ranked, relevant))
            per_query["ndcg@k"].append(ndcg_at_k(ranked, relevant, k))

        n = len(samples) or 1
        report["strategies"][strategy] = {
            "recall@k": round(sum(per_query["recall@k"]) / n, 4),
            "mrr": round(sum(per_query["mrr"]) / n, 4),
            "ndcg@k": round(sum(per_query["ndcg@k"]) / n, 4),
            "avg_latency_ms": round(sum(per_query["latency_ms"]) / n, 2),
        }
    return report
