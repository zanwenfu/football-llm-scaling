"""Generic parser-rescue taxonomy for any multi-field structured output.

The paper's closing argument is that the parser-rescue gap is likely to
surface on any benchmark that applies a "parse any reliable field"
convention to a multi-field generative output — function calling (name +
arguments), JSON output, tool use, structured summarization, ... This
module lifts the football-specific regime taxonomy in :mod:`consistency`
to an arbitrary set of fields and an arbitrary label type, so a reviewer
can drop the taxonomy into their own benchmark with a handful of lines.

Contract
--------

1. Describe each parseable channel in your output as a :class:`FieldSpec`
   with a name and an ``extract(raw) -> Optional[T]`` function.
2. Pick one field as the *primary* metric (i.e. the field your benchmark's
   main accuracy number currently uses). For football this is ``score``;
   for function calling it would typically be the argument dictionary.
3. Call :func:`classify_multifield` with the parsed values and the GT
   label. Regime semantics generalize the football A / B / C / C' / D
   taxonomy in the obvious way — see :class:`GenericRegime`.
4. Aggregate with :func:`structured_consistency_table` to get the same
   shape of table the paper reports for football.

JSON helper — :func:`extract_json_path` — makes adapting this to tool-use
benchmarks a one-liner per field.

The module is pure-python (stdlib only).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Any, Callable, Hashable, Iterable, Optional, TypeVar

# ---------------------------------------------------------------------------
# Extractor interface
# ---------------------------------------------------------------------------

T = TypeVar("T", bound=Hashable)


@dataclass(frozen=True)
class FieldSpec:
    """Describes one parseable channel of a structured output.

    ``extract`` takes the raw model output string and returns either a
    hashable label (anything ``==``-comparable) or ``None`` if the channel
    did not parse.
    """

    name: str
    extract: Callable[[str], Any]


# ---------------------------------------------------------------------------
# Generic regime taxonomy
# ---------------------------------------------------------------------------


class GenericRegime(str, Enum):
    """Generalization of the football A / B / C / C' / D taxonomy.

    Definitions (given N fields, a *primary* field, and ground truth ``g``):

    * ``A`` — every parsed field equals the others AND the primary equals ``g``.
    * ``B`` — every parsed field equals the others but the primary != ``g``.
    * ``C`` — some fields disagree, and the primary equals ``g``.
      (The parser rescues: primary was right, other channels were not.)
    * ``C_inv`` — some fields disagree, the primary != ``g``, but some other
      field equals ``g``. (The parser penalizes: primary was wrong, another
      channel had the right answer.)
    * ``D`` — some fields disagree, and no parsed field equals ``g``.
    * ``U`` — at least one field failed to parse.

    For N=2 with primary=``score`` this reduces exactly to :class:`consistency.Regime`.
    """

    COHERENT_SUCCESS = "A"
    SELF_CONSISTENT_MISTAKE = "B"
    PRIMARY_RESCUED = "C"
    PRIMARY_PENALIZED = "C_inv"
    FRAGMENTED = "D"
    UNPARSEABLE = "U"


def classify_multifield(
    fields: dict[str, Any],
    ground_truth: Any,
    primary: str,
) -> GenericRegime:
    """Classify one sample's multi-field output against ``ground_truth``.

    ``fields[name]`` is either the extracted value (any hashable) or ``None``
    for an unparseable channel. ``primary`` is the dict key whose match to
    ``ground_truth`` determines C vs. C_inv.
    """
    if primary not in fields:
        raise KeyError(f"primary field {primary!r} not in {list(fields)}")
    if any(v is None for v in fields.values()):
        return GenericRegime.UNPARSEABLE

    values = list(fields.values())
    all_equal = all(v == values[0] for v in values[1:])
    if all_equal:
        return (
            GenericRegime.COHERENT_SUCCESS
            if values[0] == ground_truth
            else GenericRegime.SELF_CONSISTENT_MISTAKE
        )

    if fields[primary] == ground_truth:
        return GenericRegime.PRIMARY_RESCUED
    if any(v == ground_truth for k, v in fields.items() if k != primary):
        return GenericRegime.PRIMARY_PENALIZED
    return GenericRegime.FRAGMENTED


# ---------------------------------------------------------------------------
# Parse a whole record, run the taxonomy, build a table
# ---------------------------------------------------------------------------


@dataclass
class ParsedRecord:
    """Per-sample parse result across a schema."""

    raw: str
    fields: dict[str, Any]
    ground_truth: Any
    regime: GenericRegime

    @property
    def primary_correct(self) -> bool:
        return self.regime in (
            GenericRegime.COHERENT_SUCCESS,
            GenericRegime.PRIMARY_RESCUED,
        )

    @property
    def coherent_correct(self) -> bool:
        return self.regime is GenericRegime.COHERENT_SUCCESS


def parse_multifield(
    raw: str,
    specs: Iterable[FieldSpec],
) -> dict[str, Any]:
    """Apply each field spec to ``raw`` and return {name: value_or_None}."""
    return {s.name: s.extract(raw) for s in specs}


@dataclass
class StructuredConsistencyTable:
    """Aggregated counts + three accuracy views over a multi-field benchmark."""

    n: int
    regime_counts: dict[str, int]
    primary_accuracy: float  # C + A
    coherence_required_accuracy: float  # A only
    primary_rescue_rate: float  # C / N
    primary_penalty_rate: float  # C_inv / N
    text_score_style_agreement: float | None = None  # only meaningful for 2-field

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "regime_counts": self.regime_counts,
            "primary_accuracy": self.primary_accuracy,
            "coherence_required_accuracy": self.coherence_required_accuracy,
            "primary_rescue_rate": self.primary_rescue_rate,
            "primary_penalty_rate": self.primary_penalty_rate,
            "text_score_style_agreement": self.text_score_style_agreement,
        }


def structured_consistency_table(
    raws_and_gts: Iterable[tuple[str, Any]],
    specs: list[FieldSpec],
    primary: str,
) -> StructuredConsistencyTable:
    """Compute the paper's table shape on any multi-field benchmark.

    ``raws_and_gts`` is an iterable of ``(raw_output, ground_truth)`` pairs.
    ``primary`` is the field name used by the benchmark's main accuracy
    metric (the one that gets "rescued" by the current eval convention).

    Returns a :class:`StructuredConsistencyTable` with:

    * ``primary_accuracy`` — fraction where the primary field == GT.
      (This is what the benchmark normally reports.)
    * ``coherence_required_accuracy`` — fraction where all fields == GT.
      (The diagnostic the paper recommends reporting alongside.)
    * ``primary_rescue_rate`` — fraction of samples in Regime C
      (disagreement where primary happens to match GT).
    * ``primary_penalty_rate`` — fraction of samples in Regime C_inv
      (disagreement where a non-primary field would have been right).

    A large gap between the first two numbers is the signature of the
    parser-rescue phenomenon on the benchmark.
    """
    counter: Counter[GenericRegime] = Counter()
    agreement_num = 0
    agreement_den = 0
    n = 0
    for raw, gt in raws_and_gts:
        n += 1
        fields = parse_multifield(raw, specs)
        regime = classify_multifield(fields, gt, primary)
        counter[regime] += 1
        if len(specs) == 2 and all(v is not None for v in fields.values()):
            agreement_den += 1
            if len(set(fields.values())) == 1:
                agreement_num += 1

    return StructuredConsistencyTable(
        n=n,
        regime_counts={r.value: counter[r] for r in GenericRegime},
        primary_accuracy=(
            counter[GenericRegime.COHERENT_SUCCESS]
            + counter[GenericRegime.PRIMARY_RESCUED]
        ) / n if n else 0.0,
        coherence_required_accuracy=
            counter[GenericRegime.COHERENT_SUCCESS] / n if n else 0.0,
        primary_rescue_rate=counter[GenericRegime.PRIMARY_RESCUED] / n if n else 0.0,
        primary_penalty_rate=counter[GenericRegime.PRIMARY_PENALIZED] / n if n else 0.0,
        text_score_style_agreement=(
            agreement_num / agreement_den if agreement_den else None
        ),
    )


# ---------------------------------------------------------------------------
# Handy extractors for common structured-output benchmarks
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_block(raw: str) -> Optional[dict]:
    """Find the first balanced JSON object in ``raw`` and parse it.

    Tries a strict greedy match first, then falls back to progressive
    trimming — enough for the common case of a JSON object embedded in a
    chatty model response. Returns ``None`` on failure.
    """
    m = _JSON_BLOCK_RE.search(raw)
    if not m:
        return None
    blob = m.group(0)
    for candidate in (blob, *_trim_from_right(blob)):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _trim_from_right(s: str, step: int = 1, max_trim: int = 200) -> Iterable[str]:
    for k in range(step, min(max_trim, len(s)), step):
        yield s[:-k]


def extract_json_path(raw: str, path: str, default: Any = None) -> Any:
    """Extract ``obj.a.b.c`` from a JSON block inside ``raw``.

    ``path`` uses dots for nested dict keys. Returns ``default`` if the JSON
    does not parse or any step in the path is missing. Lists are not
    indexed — callers that need list indexing should compose their own
    extractor with the parsed object from :func:`extract_json_block`.
    """
    obj = extract_json_block(raw)
    if obj is None:
        return default
    cur: Any = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def extract_function_name(raw: str) -> Optional[str]:
    """Get ``function.name`` from an OpenAI-style function-call block."""
    # Common shapes: {"function": {"name": "foo", "arguments": "..."}}
    # or {"tool_calls": [{"function": {"name": "foo", ...}}]}
    obj = extract_json_block(raw)
    if obj is None:
        return None
    if "function" in obj and isinstance(obj["function"], dict):
        return obj["function"].get("name")
    if "name" in obj and isinstance(obj["name"], str):
        return obj["name"]
    tcs = obj.get("tool_calls")
    if isinstance(tcs, list) and tcs:
        first = tcs[0]
        if isinstance(first, dict):
            fn = first.get("function") or {}
            return fn.get("name")
    return None


def extract_function_argument(raw: str, argname: str) -> Any:
    """Get one named argument from a function-call block. ``None`` if missing."""
    obj = extract_json_block(raw)
    if obj is None:
        return None
    # Try common shapes: {"function": {"arguments": {...}}} or
    # {"function": {"arguments": "<stringified JSON>"}}.
    fn = obj.get("function") or (obj.get("tool_calls") or [{}])[0].get("function") or {}
    args = fn.get("arguments", obj.get("arguments"))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if isinstance(args, dict):
        return args.get(argname)
    return None
