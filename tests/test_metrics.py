from app.retrieval.evaluation import mrr, ndcg_at_k, recall_at_k


def test_recall_at_k():
    ranked = [10, 20, 30]
    assert recall_at_k(ranked, {20, 40}, k=2) == 0.5
    assert recall_at_k(ranked, {99}, k=3) == 0.0


def test_mrr_first_hit():
    assert mrr([5, 7, 9], {7}) == 1 / 2
    assert mrr([5, 7, 9], {99}) == 0.0


def test_ndcg_perfect_order_equals_one():
    ranked = [1, 2, 3]
    assert ndcg_at_k(ranked, {1, 2, 3}, k=3) == 1.0


def test_ndcg_below_perfect_for_worse_rank():
    ranked = [9, 1, 2]
    score = ndcg_at_k(ranked, {1, 2}, k=3)
    assert 0 < score < 1
