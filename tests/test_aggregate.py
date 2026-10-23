"""Multi-seed aggregation + bootstrap CI."""

from __future__ import annotations

import pytest

from aggregate import aggregate_across_seeds, bootstrap_ci


def _rec(gt: str, raw: str, is_anon: bool = False):
    return {
        "is_anon": is_anon,
        "gt": {"result": gt, "home_goals": 0, "away_goals": 0, "parsed": True},
        "raw_output": raw,
    }


class TestBootstrapCI:
    def test_single_value_collapses_to_point(self):
        low, high = bootstrap_ci([0.5])
        assert low == 0.5
        assert high == 0.5

    def test_empty_is_zero_interval(self):
        assert bootstrap_ci([]) == (0.0, 0.0)

    def test_interval_contains_mean(self):
        values = [0.40, 0.55, 0.60, 0.45, 0.50]
        mean = sum(values) / len(values)
        low, high = bootstrap_ci(values, n_resamples=1000)
        assert low <= mean <= high

    def test_determinism_given_seed(self):
        values = [0.4, 0.5, 0.6]
        a = bootstrap_ci(values, seed=7)
        b = bootstrap_ci(values, seed=7)
        assert a == b

    def test_wider_spread_produces_wider_interval(self):
        narrow = bootstrap_ci([0.49, 0.50, 0.51], seed=1)
        wide = bootstrap_ci([0.30, 0.50, 0.70], seed=1)
        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


class TestAggregateAcrossSeeds:
    def test_shape_and_mean(self):
        # Seed A: 2/3 correct on score_acc
        seed_a = [
            _rec("home_win", "Prediction: home_win\nScore: 1-0"),
            _rec("away_win", "Prediction: home_win\nScore: 0-1"),
            _rec("draw", "Prediction: draw\nScore: 2-0"),  # wrong
        ]
        # Seed B: 3/3 correct
        seed_b = [
            _rec("home_win", "Prediction: home_win\nScore: 1-0"),
            _rec("away_win", "Prediction: away_win\nScore: 0-1"),
            _rec("draw", "Prediction: draw\nScore: 1-1"),
        ]
        agg = aggregate_across_seeds([seed_a, seed_b])
        assert agg.n_seeds == 2
        assert agg.score_acc.mean == pytest.approx((2 / 3 + 3 / 3) / 2)

    def test_empty_seeds_raises(self):
        with pytest.raises(ValueError):
            aggregate_across_seeds([])

    def test_single_seed_has_zero_std(self):
        seed = [_rec("home_win", "Prediction: home_win\nScore: 1-0")] * 5
        agg = aggregate_across_seeds([seed])
        assert agg.score_acc.std == 0.0
        assert agg.score_acc.ci95_low == agg.score_acc.mean
