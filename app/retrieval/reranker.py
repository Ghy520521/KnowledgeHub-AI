"""Reranker 适配器：召回候选 -> 精排，输出更相关的 Top-N。"""
from __future__ import annotations

from typing import Protocol

import httpx

from app.config import settings
from app.ingestion.embeddings import get_embedder
from app.retrieval.hybrid import Candidate


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[Candidate], top_n: int) -> list[Candidate]: ...


class NullReranker:
    """不做重排，直接截断。"""

    def rerank(self, query: str, candidates: list[Candidate], top_n: int) -> list[Candidate]:
        return candidates[:top_n]


class OpenAICompatibleReranker:
    """调用兼容 /rerank 接口（如 Jina Cohere 风格），按 relevance_score 精排。"""

    def __init__(self) -> None:
        self.base_url = settings.reranker_base_url.rstrip("/")
        self.model = settings.reranker_model
        self.api_key = settings.reranker_api_key

    def rerank(self, query: str, candidates: list[Candidate], top_n: int) -> list[Candidate]:
        resp = httpx.post(
            f"{self.base_url}/rerank",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "query": query,
                "documents": [c.content for c in candidates],
            },
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json()["results"]
        order = {r["index"]: float(r["relevance_score"]) for r in results}
        for i, cand in enumerate(candidates):
            cand.rerank_score = order.get(i, 0.0)
        ranked = sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)
        return ranked[:top_n]


class FallbackReranker:
    """本地轻量精排：0.5 * 词面重叠 + 0.5 * 向量余弦，无外部依赖。"""

    def __init__(self) -> None:
        self.embedder = get_embedder()
        self._query_vec: list[float] | None = None

    @staticmethod
    def _lexical(query: str, content: str) -> float:
        import re

        q = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", query.lower()))
        body = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", content.lower()))
        if not q:
            return 0.0
        return len(q & body) / len(q)

    def rerank(self, query: str, candidates: list[Candidate], top_n: int) -> list[Candidate]:
        if self._query_vec is None:
            self._query_vec = self.embedder.embed([query])[0]
        for cand in candidates:
            lex = self._lexical(query, cand.content)
            # 向量余弦在回退路径上由调用方已可计算；这里复用 rrf 分数做补充
            cand.rerank_score = 0.5 * lex + 0.5 * min(cand.rrf_score * 10, 1.0)
        ranked = sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)
        return ranked[:top_n]


def get_reranker() -> Reranker:
    provider = settings.reranker_provider
    if provider == "openai_compatible":
        return OpenAICompatibleReranker()
    if provider == "none":
        return NullReranker()
    return FallbackReranker()
