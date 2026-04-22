"""Regime taxonomy + disagreement directionality."""

from __future__ import annotations

from consistency import (
    Regime,
    classify,
    disagreement_directionality,
    find_examples_by_regime,
    regime_counts,
)
from parsing import Label


def _rec(gt: str, raw: str):
    return {
        "gt": {"result": gt, "home_goals": 0, "away_goals": 0, "parsed": True},
        "raw_output": raw,
    }


class TestClassify:
    def test_coherent_success(self):
        assert classify(Label.HOME_WIN, Label.HOME_WIN, Label.HOME_WIN) is Regime.COHERENT_SUCCESS

    def test_self_consistent_mistake(self):
        assert classify(Label.HOME_WIN, Label.HOME_WIN, Label.AWAY_WIN) is Regime.SELF_CONSISTENT_MISTAKE

    def test_parser_rescued(self):
        # Text says home, score says away, GT is away — the parser rescues.
        assert classify(Label.HOME_WIN, Label.AWAY_WIN, Label.AWAY_WIN) is Regime.PARSER_RESCUED

    def test_parser_penalized(self):
        # Text says home, score says away, GT is home — the parser penalizes.
        assert classify(Label.HOME_WIN, Label.AWAY_WIN, Label.HOME_WIN) is Regime.PARSER_PENALIZED

    def test_fragmented(self):
        # Text says home, score says away, GT is draw — neither matches.
        assert classify(Label.HOME_WIN, Label.AWAY_WIN, Label.DRAW) is Regime.FRAGMENTED

    def test_unparseable_when_channel_missing(self):
        assert classify(None, Label.HOME_WIN, Label.HOME_WIN) is Regime.UNPARSEABLE
        assert classify(Label.HOME_WIN, None, Label.HOME_WIN) is Regime.UNPARSEABLE
        assert classify(Label.HOME_WIN, Label.HOME_WIN, None) is Regime.UNPARSEABLE


class TestRegimeCounts:
    def test_counts_match_hand_computed(self):
        recs = [
            _rec("home_win", "Prediction: home_win\nScore: 2-1"),  # A
            _rec("away_win", "Prediction: home_win\nScore: 0-2"),  # C (parser-rescued)
            _rec("home_win", "Prediction: home_win\nScore: 0-2"),  # C_inv (parser-penalized)
            _rec("draw", "Prediction: home_win\nScore: 2-0"),      # B (both home, GT draw)
            _rec("draw", "Prediction: home_win\nScore: 0-2"),      # D (neither matches)
        ]
        rc = regime_counts(recs)
        assert rc.A == 1
        assert rc.B == 1
        assert rc.C == 1
        assert rc.C_inv == 1
        assert rc.D == 1
        assert rc.total == 5

    def test_parser_rescue_rate(self):
        recs = [_rec("away_win", "Prediction: home_win\nScore: 0-2")] * 3 + [
            _rec("home_win", "Prediction: home_win\nScore: 2-1")
        ]
        assert regime_counts(recs).parser_rescue_rate == 0.75


class TestDisagreementDirectionality:
    def test_ratio_is_above_one_when_parser_rescues(self):
        # Two C + one C_inv -> ratio 2.0 (rescue > penalty)
        recs = [
            _rec("away_win", "Prediction: home_win\nScore: 0-2"),  # C
            _rec("away_win", "Prediction: home_win\nScore: 0-2"),  # C
            _rec("home_win", "Prediction: home_win\nScore: 0-2"),  # C_inv
        ]
        d = disagreement_directionality(recs)
        assert d.score_matches_gt == 2
        assert d.text_matches_gt == 1
        assert d.parser_rescue_to_penalty_ratio == 2.0

    def test_ratio_below_one_when_parser_penalizes(self):
        # This is the qualitative pattern the paper finds for ICL and CoT.
        recs = [
            _rec("home_win", "Prediction: home_win\nScore: 0-2"),  # C_inv
            _rec("home_win", "Prediction: home_win\nScore: 0-2"),  # C_inv
            _rec("away_win", "Prediction: home_win\nScore: 0-2"),  # C
        ]
        d = disagreement_directionality(recs)
        assert 0.0 < d.parser_rescue_to_penalty_ratio < 1.0

    def test_zero_disagreements_when_everything_coheres(self):
        recs = [_rec("home_win", "Prediction: home_win\nScore: 2-1")] * 3
        d = disagreement_directionality(recs)
        assert d.n_disagreements == 0


class TestFindExamples:
    def test_limit_is_respected(self):
        recs = [_rec("away_win", "Prediction: home_win\nScore: 0-2")] * 10
        assert len(find_examples_by_regime(recs, Regime.PARSER_RESCUED, limit=3)) == 3
