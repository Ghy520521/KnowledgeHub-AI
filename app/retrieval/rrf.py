"""Reciprocal Rank Fusion：不融合不同量纲的相似度分数，只融合排名。

score(c) = Σ_r 1 / (k + rank_r(c))
不同检索器返回的原始分数量纲不同（余弦距离 vs ts_rank），直接相加没有意义，
RRF 只利用各检索器的排名，对量纲不敏感，实现简单且鲁棒。
"""
from __future__ import annotations


def rrf_fuse(rank_lists: list[list[tuple[int, float]],], k: int = 60) -> list[tuple[int, float]]:
    """输入：多个检索器的 [(chunk_id, raw_score), ...]（已按相关度降序）。

    输出：[(chunk_id, rrf_score), ...]，按 RRF 分数降序。
    """
    scores: dict[int, float] = {}
    for lst in rank_lists:
        for rank, (chunk_id, _raw_score) in enumerate(lst, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
