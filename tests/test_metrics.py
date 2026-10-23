"""Accuracy, CI, and distribution metrics."""

from __future__ import annotations

import math

import pytest

from metrics import (
    always_home_accuracy,
    compute_all_splits,
    compute_metrics,
    metrics_to_legacy_schema,
    wilson_ci,
    wilson_halfwidth_pp,
)


def _rec(gt_result: str, gt_h: int, gt_a: int, raw: str, is_anon: bool = False):
    return {
        "is_anon": is_anon,
        "gt": {"result": gt_result, "home_goals": gt_h, "away_goals": gt_a, "parsed": True},
        "raw_output": raw,
    }


class TestWilsonCI:
    def test_zero_samples_returns_zero_interval(self):
        assert wilson_ci(0, 0) == (0.0, 0.0)

    def test_half_rate_has_symmetric_interval(self):
        low, high = wilson_ci(50, 100)
        assert 0.40 < low < 0.51
        assert 0.49 < high < 0.60

    def test_boundary_zero_and_one(self):
        low, high = wilson_ci(0, 10)
        assert low == 0.0
        assert 0 < high < 1
        low, high = wilson_ci(10, 10)
        assert 0 < low < 1
        assert high == pytest.approx(1.0)

    def test_halfwidth_pp_matches_paper_precision(self):
        # Paper reports ±8.5 pp for ICL at n=128 with p=0.492 (k=63):
        hw = wilson_halfwidth_pp(63, 128)
        assert 8.3 < hw < 8.7


class TestComputeMetrics:
    def test_three_accuracy_views_diverge(self):
        # One parser-rescue case, one coherent correct, one wrong.
        recs = [
            _rec("away_win", 0, 2, "Prediction: home_win\nScore: 0-2"),  # score rescues
            _rec("home_win", 2, 0, "Prediction: home_win\nScore: 2-0"),  # coherent ✓
            _rec("draw", 1, 1, "Prediction: away_win\nScore: 3-0"),      # coherent ✗
        ]
        m = compute_metrics(recs)
        # score overrides text: 2/3 correct (row 1 score=away, row 2 score=home; row 3 score=home≠draw)
        assert m.score_acc == pytest.approx(2 / 3)
        # text only: 1/3 correct (row 2)
        assert m.text_acc == pytest.approx(1 / 3)
        # coherence-required: 1/3 correct (row 2)
        assert m.coherence_acc == pytest.approx(1 / 3)

    def test_always_home_accuracy(self):
        recs = [
            _rec("home_win", 1, 0, "Prediction: home_win\nScore: 1-0"),
            _rec("home_win", 2, 1, "Prediction: home_win\nScore: 2-1"),
            _rec("away_win", 0, 1, "Prediction: away_win\nScore: 0-1"),
        ]
        assert always_home_accuracy(recs) == pytest.approx(2 / 3)

    def test_splits_are_disjoint_and_sum_to_overall(self):
        recs = [
            _rec("home_win", 1, 0, "Prediction: home_win\nScore: 1-0", is_anon=False),
            _rec("away_win", 0, 1, "Prediction: away_win\nScore: 0-1", is_anon=True),
        ]
        splits = compute_all_splits(recs)
        assert splits["overall"].n == splits["named"].n + splits["anon"].n

    def test_legacy_schema_roundtrip_preserves_pred_dist_totals(self):
        recs = [
            _rec("home_win", 1, 0, "Prediction: home_win\nScore: 1-0"),
            _rec("away_win", 0, 1, "Prediction: away_win\nScore: 0-1"),
            _rec("draw", 0, 0, "Prediction: draw\nScore: 0-0"),
        ]
        legacy = metrics_to_legacy_schema(compute_metrics(recs))
        counts = legacy["pred_dist"]
        assert counts["home_win"] + counts["away_win"] + counts["draw"] + counts["other"] == legacy["total"]

    def test_goal_mae(self):
        recs = [
            _rec("home_win", 2, 0, "Prediction: home_win\nScore: 2-0"),  # 0 error
            _rec("draw", 1, 1, "Prediction: draw\nScore: 2-2"),          # 2/2 = 1.0
        ]
        m = compute_metrics(recs)
        # (0 + 0 + 1 + 1) / 4 = 0.5
        assert m.goal_mae == pytest.approx(0.5)


class TestAgreement:
    def test_full_agreement_when_coherent(self):
        recs = [_rec("home_win", 2, 1, "Prediction: home_win\nScore: 2-1")]
        assert compute_metrics(recs).text_score_agreement == 1.0

    def test_agreement_below_one_when_fields_decouple(self):
        recs = [
            _rec("home_win", 2, 1, "Prediction: home_win\nScore: 0-2"),
            _rec("home_win", 2, 1, "Prediction: home_win\nScore: 2-1"),
        ]
        # 1/2 = 0.5
        assert compute_metrics(recs).text_score_agreement == pytest.approx(0.5)

    def test_agreement_is_nan_when_nothing_parses(self):
        recs = [_rec("home_win", 1, 0, "nothing parseable here.")]
        assert math.isnan(compute_metrics(recs).text_score_agreement)
