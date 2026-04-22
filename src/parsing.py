"""Three independent extractors for a single model output.

The paper's central claim is that fine-tuned outputs carry three separable
"channels" — the text label on the `Prediction:` line, the numeric score on the
`Score:` line, and the argument on the `Reasoning:` line — and that small-data
fine-tuning fails to bind them into a coherent commitment. This module provides
the three extractors and a single `parse_output` that returns all three.

The extractors are deliberately *independent*: `extract_text_label` ignores the
score line, `extract_score` ignores the prediction line. The legacy convention
of "use the score when both parse" is isolated in
`resolve_score_overrides_text` so that the analysis code can measure the size
of the effect instead of silently absorbing it.

All functions are pure and import-cheap (stdlib only).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


class Label(str, Enum):
    """The three possible match outcomes."""

    HOME_WIN = "home_win"
    AWAY_WIN = "away_win"
    DRAW = "draw"

    @classmethod
    def from_str(cls, s: str) -> Optional["Label"]:
        s = s.strip().lower().replace("-", "_").replace(" ", "_")
        if s in ("home_win", "home", "homewin", "home_draw_home"):
            return cls.HOME_WIN
        if s in ("away_win", "away", "awaywin"):
            return cls.AWAY_WIN
        if s in ("draw", "tie", "home_draw"):
            return cls.DRAW
        return None

    @classmethod
    def from_score(cls, home_goals: int, away_goals: int) -> "Label":
        if home_goals > away_goals:
            return cls.HOME_WIN
        if away_goals > home_goals:
            return cls.AWAY_WIN
        return cls.DRAW


# ---------------------------------------------------------------------------
# Text-label extractor
# ---------------------------------------------------------------------------

_TEXT_LABEL_RE = re.compile(r"prediction\s*[:\-]\s*([a-zA-Z_][\w\s\-]*)", re.IGNORECASE)


def extract_text_label(raw: str) -> Optional[Label]:
    """Return the label written on the `Prediction:` line, or None.

    This extractor ignores everything on the `Score:` line. It is the
    "what the model says" view used by the paper's `text_acc` column.
    """
    if not raw:
        return None
    m = _TEXT_LABEL_RE.search(raw)
    if not m:
        return None
    token = m.group(1).split()[0]
    return Label.from_str(token)


# ---------------------------------------------------------------------------
# Score extractor (tail-scan, with realism check)
# ---------------------------------------------------------------------------

# The bare-digit fallback kept matching squad-goal totals like "303-167" in
# CoT reasoning paragraphs. The paper's hardened parser restricts the fallback
# to the tail of the output and requires both sides to be realistic match
# scores (<=15 goals). See `notebooks/03b_cot_rerun.ipynb` for the fix.
_SCORE_LABELED_RE = re.compile(r"score\s*[:\-]\s*(\d{1,2})\s*[-–]\s*(\d{1,2})", re.IGNORECASE)
_SCORE_BARE_RE = re.compile(r"(\d{1,2})\s*[-–]\s*(\d{1,2})")
_REALISTIC_SCORE_MAX = 15
_TAIL_SCAN_CHARS = 400


def extract_score(raw: str) -> Optional[tuple[int, int]]:
    """Return `(home_goals, away_goals)` from the `Score:` line, or None.

    Strategy: prefer the `Score: h-a` labeled form anywhere in the output.
    If absent, fall back to a bare `h-a` pattern but restricted to the last
    `_TAIL_SCAN_CHARS` characters (where a concluding score would appear) and
    requiring both sides <= 15. This prevents aggregate statistics like
    `303-167 squad goals` from being mis-parsed as match scores, which was
    the dominant parse error in the original CoT evaluation.
    """
    if not raw:
        return None
    m = _SCORE_LABELED_RE.search(raw)
    if m:
        h, a = int(m.group(1)), int(m.group(2))
        if _is_realistic(h, a):
            return (h, a)
    tail = raw[-_TAIL_SCAN_CHARS:]
    for match in _SCORE_BARE_RE.finditer(tail):
        h, a = int(match.group(1)), int(match.group(2))
        if _is_realistic(h, a):
            return (h, a)
    return None


def _is_realistic(h: int, a: int) -> bool:
    return 0 <= h <= _REALISTIC_SCORE_MAX and 0 <= a <= _REALISTIC_SCORE_MAX


def extract_score_label(raw: str) -> Optional[Label]:
    """Label implied by the numeric `Score:` line, computed independently."""
    score = extract_score(raw)
    if score is None:
        return None
    return Label.from_score(*score)


# ---------------------------------------------------------------------------
# Reasoning-paragraph label extractor (heuristic, documented as noisy)
# ---------------------------------------------------------------------------

_REASONING_BLOCK_RE = re.compile(r"reasoning\s*[:\-]\s*(.*)", re.IGNORECASE | re.DOTALL)

_HOME_KEYWORDS = (
    "home team",
    "home side",
    "host",
    "team a ",
    "first team",
    "home_win",
)
_AWAY_KEYWORDS = (
    "away team",
    "visiting",
    "visitor",
    "team b ",
    "second team",
    "away_win",
)
_DRAW_KEYWORDS = (
    "even match",
    "balanced",
    "draw",
    "tie ",
    "level game",
)


def extract_reasoning_label(raw: str) -> Optional[Label]:
    """Infer the intended winner from the `Reasoning:` paragraph.

    This is a keyword heuristic and is known to be noisy (low hit rate:
    ~13–50/128 samples per condition in the paper's consistency analysis). It
    is exposed for completeness and for debugging; the primary channels in the
    paper are text and score.
    """
    if not raw:
        return None
    m = _REASONING_BLOCK_RE.search(raw)
    if not m:
        return None
    text = m.group(1).lower()
    hits = {
        Label.HOME_WIN: sum(text.count(k) for k in _HOME_KEYWORDS),
        Label.AWAY_WIN: sum(text.count(k) for k in _AWAY_KEYWORDS),
        Label.DRAW: sum(text.count(k) for k in _DRAW_KEYWORDS),
    }
    top = max(hits.items(), key=lambda kv: kv[1])
    if top[1] == 0:
        return None
    return top[0]


# ---------------------------------------------------------------------------
# Unified parse result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedOutput:
    """All three channels extracted from one raw model output."""

    raw: str
    text_label: Optional[Label]
    score_label: Optional[Label]
    home_goals: Optional[int]
    away_goals: Optional[int]
    reasoning_label: Optional[Label]

    @property
    def has_prediction_line(self) -> bool:
        return self.text_label is not None

    @property
    def has_score_line(self) -> bool:
        return self.score_label is not None

    @property
    def is_coherent(self) -> bool:
        """True iff text and score label are both parsed and agree."""
        return (
            self.text_label is not None
            and self.score_label is not None
            and self.text_label == self.score_label
        )


def parse_output(raw: str) -> ParsedOutput:
    """Extract all three channels from a single raw model output."""
    score = extract_score(raw)
    score_label = Label.from_score(*score) if score is not None else None
    return ParsedOutput(
        raw=raw,
        text_label=extract_text_label(raw),
        score_label=score_label,
        home_goals=score[0] if score else None,
        away_goals=score[1] if score else None,
        reasoning_label=extract_reasoning_label(raw),
    )


# ---------------------------------------------------------------------------
# Legacy "score overrides text" resolution
# ---------------------------------------------------------------------------


def resolve_score_overrides_text(parsed: ParsedOutput) -> Optional[Label]:
    """Apply the legacy eval convention: use score when both exist, else text.

    This matches the original `eval_harness.ipynb` logic and is the number
    reported in the paper's `score_acc` column. It is the *parser-rescue*
    convention — the point of the paper is that it silently changes the
    reported accuracy of small-data fine-tuned models by up to 20 pp.
    """
    if parsed.score_label is not None:
        return parsed.score_label
    return parsed.text_label
