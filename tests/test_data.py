"""Nested stratified prefix sampling — the scaling ablation's data split."""

from __future__ import annotations

from collections import Counter

import pytest

from data import stratified_nested_prefix_indices


def _mock_train(n_home: int = 80, n_away: int = 70, n_draw: int = 50):
    recs = (
        [{"metadata": {"result": "home_win"}} for _ in range(n_home)]
        + [{"metadata": {"result": "away_win"}} for _ in range(n_away)]
        + [{"metadata": {"result": "draw"}} for _ in range(n_draw)]
    )
    return recs


class TestStratifiedNestedPrefix:
    def test_nested_subset_invariant(self):
        train = _mock_train()
        idx = stratified_nested_prefix_indices(train, (48, 96, 192))
        assert set(idx[48]).issubset(set(idx[96]))
        assert set(idx[96]).issubset(set(idx[192]))

    def test_sizes_are_exact(self):
        train = _mock_train()
        idx = stratified_nested_prefix_indices(train, (48, 96, 192))
        assert len(idx[48]) == 48
        assert len(idx[96]) == 96
        assert len(idx[192]) == 192

    def test_class_balance_is_preserved(self):
        # With 80/70/50 = 200 total: home~40%, away~35%, draw~25%.
        # At n=48 we expect roughly 19/17/12 ± 1.
        train = _mock_train(80, 70, 50)
        idx = stratified_nested_prefix_indices(train, (48,))
        counts: Counter[str] = Counter()
        for i in idx[48]:
            counts[train[i]["metadata"]["result"]] += 1
        assert abs(counts["home_win"] - 19) <= 2
        assert abs(counts["away_win"] - 17) <= 2
        assert abs(counts["draw"] - 12) <= 2

    def test_determinism(self):
        train = _mock_train()
        a = stratified_nested_prefix_indices(train, (96,), seed=42)
        b = stratified_nested_prefix_indices(train, (96,), seed=42)
        assert a[96] == b[96]

    def test_different_seed_produces_different_selection(self):
        train = _mock_train()
        a = stratified_nested_prefix_indices(train, (96,), seed=1)
        b = stratified_nested_prefix_indices(train, (96,), seed=2)
        assert a[96] != b[96]

    def test_budget_exceeding_total_raises(self):
        train = _mock_train(5, 5, 5)
        with pytest.raises(ValueError):
            stratified_nested_prefix_indices(train, (100,))
