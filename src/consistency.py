"""The parser-rescue gap: regime taxonomy and disagreement directionality.

The paper's central diagnostic is the four-way regime classification of each
output based on whether the text label, the score label, and the ground-truth
label all agree. This module implements the taxonomy and two derived tables:

* **Regime counts per condition** (Section 3.2, "Regime taxonomy").
* **Disagreement directionality** (Section 3.2, Table 2): among samples where
  text and score disagree, which side matches ground truth?

Both functions accept the raw-record list from ``results/raw/*.json``. They
depend only on :mod:`parsing` and are fast enough
(≲1 ms per 128-sample condition) to recompute on every run.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable, Optional

from parsing import Label, parse_output

# ---------------------------------------------------------------------------
# Regime taxonomy
# ---------------------------------------------------------------------------


class Regime(str, Enum):
    """Per-sample output classification used in Table 2 of the paper."""

    COHERENT_SUCCESS = "A"  # text == score == GT
    SELF_CONSISTENT_MISTAKE = "B"  # text == score != GT
    PARSER_RESCUED = "C"  # text != score, score == GT  (the key failure mode)
    PARSER_PENALIZED = "C_inv"  # text != score, text == GT
    FRAGMENTED = "D"  # text != score, neither matches GT
    UNPARSEABLE = "U"  # either channel missing

    def description(self) -> str:
        return _REGIME_DESCRIPTIONS[self]


_REGIME_DESCRIPTIONS: dict[Regime, str] = {
    Regime.COHERENT_SUCCESS: "text == score == GT — genuine coherent success",
    Regime.SELF_CONSISTENT_MISTAKE: "text == score != GT — honest, self-consistent mistake",
    Regime.PARSER_RESCUED: "text != score; score matches GT — parser rescues the model",
    Regime.PARSER_PENALIZED: "text != score; text matches GT — parser penalizes the model",
    Regime.FRAGMENTED: "text != score and neither matches GT — fragmented output",
    Regime.UNPARSEABLE: "at least one of text or score did not parse",
}


def classify(
    text_label: Optional[Label],
    score_label: Optional[Label],
    gt_label: Optional[Label],
) -> Regime:
    """Assign a sample to one of the regimes above."""
    if text_label is None or score_label is None or gt_label is None:
        return Regime.UNPARSEABLE
    if text_label == score_label == gt_label:
        return Regime.COHERENT_SUCCESS
    if text_label == score_label:
        return Regime.SELF_CONSISTENT_MISTAKE
    if score_label == gt_label:
        return Regime.PARSER_RESCUED
    if text_label == gt_label:
        return Regime.PARSER_PENALIZED
    return Regime.FRAGMENTED


# ---------------------------------------------------------------------------
# Dataclasses for summary tables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeCounts:
    """Per-condition counts of each regime (matches `final_tables.json`)."""

    A: int = 0
    B: int = 0
    C: int = 0
    C_inv: int = 0
    D: int = 0
    U: int = 0

    @property
    def total(self) -> int:
        return self.A + self.B + self.C + self.C_inv + self.D + self.U

    @property
    def parser_rescue_rate(self) -> float:
        """Fraction of all samples where the parser converted a miss to a hit."""
        n = self.total
        return self.C / n if n else 0.0

    @property
    def coherent_success_rate(self) -> float:
        n = self.total
        return self.A / n if n else 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "total": self.total,
            "parser_rescue_rate": self.parser_rescue_rate,
            "coherent_success_rate": self.coherent_success_rate,
        }


@dataclass(frozen=True)
class DisagreementDirectionality:
    """Table 2 of the paper: on text≠score disagreements, who matches GT?"""

    n_disagreements: int
    score_matches_gt: int  # Regime C
    text_matches_gt: int  # Regime C_inv
    neither_matches_gt: int  # Regime D

    @property
    def parser_rescue_to_penalty_ratio(self) -> float:
        """> 1 means the parser rescues; < 1 means the parser penalizes.

        For ICL/CoT this is < 1 (text is the more reliable channel). For
        QLoRA n>=96 it is 6–14× in the paper.
        """
        if self.text_matches_gt == 0:
            return float("inf") if self.score_matches_gt > 0 else 0.0
        return self.score_matches_gt / self.text_matches_gt

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "parser_rescue_to_penalty_ratio": self.parser_rescue_to_penalty_ratio,
        }


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def regime_counts(records: Iterable[dict[str, Any]]) -> RegimeCounts:
    """Count regimes over an iterable of result records."""
    counter: Counter[Regime] = Counter()
    for r in records:
        p = parse_output(r.get("raw_output", ""))
        gt = Label.from_str((r.get("gt") or {}).get("result", ""))
        counter[classify(p.text_label, p.score_label, gt)] += 1
    return RegimeCounts(
        A=counter[Regime.COHERENT_SUCCESS],
        B=counter[Regime.SELF_CONSISTENT_MISTAKE],
        C=counter[Regime.PARSER_RESCUED],
        C_inv=counter[Regime.PARSER_PENALIZED],
        D=counter[Regime.FRAGMENTED],
        U=counter[Regime.UNPARSEABLE],
    )


def disagreement_directionality(
    records: Iterable[dict[str, Any]],
) -> DisagreementDirectionality:
    """Paper Table 2: of text≠score disagreements, how many match GT where?"""
    rc = regime_counts(records)
    n_disagreements = rc.C + rc.C_inv + rc.D
    return DisagreementDirectionality(
        n_disagreements=n_disagreements,
        score_matches_gt=rc.C,
        text_matches_gt=rc.C_inv,
        neither_matches_gt=rc.D,
    )


# ---------------------------------------------------------------------------
# Example extraction for the appendix
# ---------------------------------------------------------------------------


def find_examples_by_regime(
    records: list[dict[str, Any]],
    regime: Regime,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return up to ``limit`` records that fall into ``regime``.

    Used by `scripts/04_run_consistency_analysis.py` to populate
    `results/examples/example_contradictions.md`.
    """
    out: list[dict[str, Any]] = []
    for r in records:
        p = parse_output(r.get("raw_output", ""))
        gt = Label.from_str((r.get("gt") or {}).get("result", ""))
        if classify(p.text_label, p.score_label, gt) == regime:
            out.append(r)
            if len(out) >= limit:
                break
    return out
