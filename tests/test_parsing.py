"""Parser regression tests.

Every test here corresponds to a specific failure mode the paper's evaluation
hit during development. The comments reference the notebook the bug was
discovered in.
"""

from __future__ import annotations

import pytest

from parsing import (
    Label,
    extract_reasoning_label,
    extract_score,
    extract_score_label,
    extract_text_label,
    parse_output,
    resolve_score_overrides_text,
)

# ---------------------------------------------------------------------------
# Text-label extractor
# ---------------------------------------------------------------------------


class TestExtractTextLabel:
    def test_canonical_three_line_output(self):
        raw = "Prediction: home_win\nScore: 2-1\nReasoning: ..."
        assert extract_text_label(raw) is Label.HOME_WIN

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("home_win", Label.HOME_WIN),
            ("away_win", Label.AWAY_WIN),
            ("draw", Label.DRAW),
            ("HOME_WIN", Label.HOME_WIN),  # case-insensitive
            ("home-win", Label.HOME_WIN),  # hyphen variant
            ("tie", Label.DRAW),  # common CoT synonym
        ],
    )
    def test_label_variants(self, label, expected):
        assert extract_text_label(f"Prediction: {label}\nScore: 1-0") is expected

    def test_home_draw_typo_is_coerced_to_draw(self):
        # Observed in icl_predictions[0]: "Prediction: home_draw"
        assert extract_text_label("Prediction: home_draw\nScore: 1-1") is Label.DRAW

    def test_no_prediction_line_returns_none(self):
        assert extract_text_label("I think the home team wins.") is None

    def test_empty_input(self):
        assert extract_text_label("") is None
        assert extract_text_label(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Score extractor — the regression the hardened parser targets
# ---------------------------------------------------------------------------


class TestExtractScore:
    def test_canonical_labeled_score(self):
        assert extract_score("Prediction: home_win\nScore: 2-1") == (2, 1)

    def test_labeled_score_with_em_dash(self):
        assert extract_score("Score: 3–0") == (3, 0)

    def test_tail_scan_without_explicit_label(self):
        # Some CoT outputs drop the "Score:" prefix in the conclusion.
        assert extract_score("...final prediction: a narrow 1-0 win.") == (1, 0)

    def test_squad_goal_totals_are_not_mis_parsed(self):
        # Bug from notebooks/03_icl_cot_baselines.ipynb: the bare-digit fallback
        # matched "303-167" in the reasoning paragraph. The hardened parser in
        # 03b_cot_rerun restricts the fallback to realistic scores (<=15 each)
        # and only in the tail of the output.
        raw = (
            "Qatar's squad output is 303 vs Ecuador's 167. "
            "Based on these stats, a draw is likely."
        )
        assert extract_score(raw) is None

    def test_score_label_picks_correct_winner(self):
        assert extract_score_label("Score: 0-2") is Label.AWAY_WIN
        assert extract_score_label("Score: 2-0") is Label.HOME_WIN
        assert extract_score_label("Score: 1-1") is Label.DRAW

    def test_score_too_large_is_rejected(self):
        assert extract_score("Score: 50-20") is None

    def test_tail_scan_respects_window(self):
        # 500-char noise followed by a score outside the 400-char tail window
        # should only be matched via the (still-absent) labeled form, not via
        # the bare fallback.
        raw = "1-0 prediction. " + "x" * 600
        assert extract_score(raw) is None


# ---------------------------------------------------------------------------
# Unified parser + coherence property
# ---------------------------------------------------------------------------


class TestParseOutput:
    def test_coherent_output(self):
        p = parse_output("Prediction: home_win\nScore: 2-1\nReasoning: ...")
        assert p.text_label is Label.HOME_WIN
        assert p.score_label is Label.HOME_WIN
        assert p.is_coherent is True

    def test_the_parser_rescue_case(self):
        # This is the canonical failure mode the paper is about.
        p = parse_output(
            "Prediction: home_win\nScore: 0-2\nReasoning: Team B has better stats."
        )
        assert p.text_label is Label.HOME_WIN
        assert p.score_label is Label.AWAY_WIN
        assert p.is_coherent is False

    def test_resolve_score_overrides_text(self):
        p = parse_output("Prediction: home_win\nScore: 0-2")
        # The legacy convention uses the score when both exist.
        assert resolve_score_overrides_text(p) is Label.AWAY_WIN

    def test_resolve_falls_back_to_text_when_score_missing(self):
        p = parse_output("Prediction: draw")
        assert resolve_score_overrides_text(p) is Label.DRAW


# ---------------------------------------------------------------------------
# Reasoning-paragraph heuristic (noisy; only sanity checks)
# ---------------------------------------------------------------------------


class TestExtractReasoningLabel:
    def test_home_mention_detected(self):
        raw = "Reasoning: The home team has stronger stats overall."
        assert extract_reasoning_label(raw) is Label.HOME_WIN

    def test_away_mention_detected(self):
        raw = "Reasoning: The away team's offense is decisive."
        assert extract_reasoning_label(raw) is Label.AWAY_WIN

    def test_no_signal_returns_none(self):
        raw = "Reasoning: statistics are inconclusive."
        # No keyword → None (the heuristic is documented as noisy).
        assert extract_reasoning_label(raw) is None
