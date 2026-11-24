from app.retrieval.rrf import rrf_fuse


def test_rrf_prefers_item_present_in_both_lists():
    # A: c1 第一；B: c2 第一，c1 第二
    fused = rrf_fuse([[(1, 0.9), (2, 0.7)], [(2, 0.8), (1, 0.6)]], k=60)
    top = fused[0][0]
    assert top == 1  # c1 在两个列表都出现，融合分更高


def test_rrf_order_is_descending():
    fused = rrf_fuse([[(1, 0.9), (2, 0.7), (3, 0.5)], [(2, 0.8)]], k=60)
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_ignores_raw_score_scale():
    # 原始分量纲不同（0.9 vs 100），RRF 只看排名
    fused = rrf_fuse([[(1, 0.9)], [(2, 100.0)]], k=60)
    assert {cid for cid, _ in fused} == {1, 2}
