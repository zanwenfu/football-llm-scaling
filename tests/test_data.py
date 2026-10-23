"""Nested stratified prefix sampling — the scaling ablation's data split."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from config import EVAL_DATA, TRAIN_DATA
from data import load_records, stratified_nested_prefix_indices

REPO = Path(__file__).resolve().parents[1]


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


class TestBundledData:
    def test_train_split_is_192_matches_named_and_anonymized(self):
        train = load_records(REPO / TRAIN_DATA)
        assert len(train) == 384
        assert Counter(r["is_anon"] for r in train) == {True: 192, False: 192}

    def test_eval_split_is_the_2022_world_cup_twice(self):
        evals = load_records(REPO / EVAL_DATA, split="eval")
        assert len(evals) == 128
        assert Counter(r["is_anon"] for r in evals) == {True: 64, False: 64}
        assert {r["metadata"]["world_cup_year"] for r in evals} == {2022}

    def test_scores_become_goal_counts(self):
        for rec in load_records(REPO / EVAL_DATA, split="eval"):
            meta = rec["metadata"]
            assert f"{meta['home_goals']}-{meta['away_goals']}" == meta["score"]

    def test_named_split_supports_every_scaling_budget(self):
        named = [r for r in load_records(REPO / TRAIN_DATA) if not r["is_anon"]]
        idx = stratified_nested_prefix_indices(named, (48, 96, 192), seed=42)
        assert set(idx[48]) <= set(idx[96]) <= set(idx[192])
        assert len(idx[192]) == 192


def _notebook_splits(full_train):
    """The draw in notebooks/04_scaling_ablation.ipynb, cells 6 and 7, verbatim in logic."""
    import random
    import re
    from collections import defaultdict

    train_named = [
        s for s in full_train
        if not ("Team A" in s["messages"][1]["content"] and "Team B" in s["messages"][1]["content"])
    ]

    def gt_label(sample):
        text = sample["messages"][2]["content"].lower()
        m = re.search(r"(\d+)\s*[-–]\s*(\d+)", text)
        if m:
            h, a = int(m.group(1)), int(m.group(2))
            return "home_win" if h > a else "away_win" if h < a else "draw"
        return next((lbl for lbl in ["home_win", "away_win", "draw"] if lbl in text), None)

    buckets = defaultdict(list)
    for i, s in enumerate(train_named):
        lbl = gt_label(s)
        if lbl:
            buckets[lbl].append(i)
    random.seed(42)
    shuffled = {k: random.sample(v, len(v)) for k, v in buckets.items()}
    total = sum(len(v) for v in shuffled.values())

    def take_prefix(n_total):
        picks = []
        for idxs in shuffled.values():
            picks.extend(idxs[: round(n_total * len(idxs) / total)])
        picks = sorted(set(picks))
        while len(picks) < n_total:
            for lbl in sorted(shuffled, key=lambda x: -len(shuffled[x])):
                for i in shuffled[lbl]:
                    if i not in picks:
                        picks.append(i)
                        break
                if len(picks) >= n_total:
                    break
        return sorted(picks[:n_total])

    return train_named, {n: take_prefix(n) for n in (48, 96, 192)}


class TestSameDrawAsTheNotebook:
    def test_library_picks_the_notebook_s_matches(self):
        train = load_records(REPO / TRAIN_DATA)
        nb_named, nb_splits = _notebook_splits(train)
        named = [r for r in train if not r["is_anon"]]
        assert [r["metadata"]["fixture_id"] for r in named] == [r["metadata"]["fixture_id"] for r in nb_named]
        assert stratified_nested_prefix_indices(named, (48, 96, 192), seed=42) == nb_splits
