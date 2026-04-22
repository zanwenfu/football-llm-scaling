"""Demo: apply the parser-rescue taxonomy to non-football benchmarks.

Runs three worked examples showing the paper's taxonomy is framework-agnostic:

1. **Football** — re-derive the paper's numbers using the generic machinery.
   Sanity check: the numbers must match :file:`results/tables/consistency_table.json`.
2. **Function calling** — a synthetic OpenAI-style tool-use benchmark where
   the model's ``function.name`` and ``function.arguments.city`` are the two
   fields. Exposes the same rescue/penalty directionality on a JSON schema.
3. **Structured JSON output** — three-field case (``intent``, ``slot1``,
   ``slot2``). Shows the generalized Regime C on a three-channel output.

The synthetic outputs are constructed to surface every regime, so a reader
can see the taxonomy's behaviour on each kind of benchmark in ~1 second.

Run::

    python scripts/demo_structured_rescue.py
"""

from __future__ import annotations

import json
from pathlib import Path

from data import load_predictions
from parsing import extract_score_label, extract_text_label
from structured import (
    FieldSpec,
    extract_function_argument,
    extract_function_name,
    extract_json_path,
    structured_consistency_table,
)


def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _print_table(t) -> None:
    counts = t.regime_counts
    print(f"  n={t.n}  regimes: A={counts['A']:3d}  B={counts['B']:3d}  "
          f"C={counts['C']:3d}  C'={counts['C_inv']:3d}  "
          f"D={counts['D']:3d}  U={counts['U']:3d}")
    print(f"  primary_accuracy              = {t.primary_accuracy:.3f}  "
          f"(what the benchmark normally reports)")
    print(f"  coherence_required_accuracy   = {t.coherence_required_accuracy:.3f}  "
          f"(the diagnostic we recommend)")
    gap = (t.primary_accuracy - t.coherence_required_accuracy) * 100
    print(f"  parser_rescue_rate            = {t.primary_rescue_rate:.3f}  "
          f"(share of samples in Regime C)")
    print(f"  primary − coherence           = {gap:+.1f} pp  "
          f"(magnitude of the gap)")


# ---------------------------------------------------------------------------
# 1) Football, via the generic machinery
# ---------------------------------------------------------------------------


def demo_football() -> None:
    _banner("1. Football (via the generic machinery) — sanity check vs. paper")
    paths = {
        "ICL": "results/raw/icl_predictions.json",
        "n192 (QLoRA)": "results/raw/scaling_predictions_n192.json",
    }
    specs = [
        FieldSpec("text", lambda raw: _enum_value(extract_text_label(raw))),
        FieldSpec("score", lambda raw: _enum_value(extract_score_label(raw))),
    ]
    for cond, path in paths.items():
        if not Path(path).exists():
            continue
        recs = load_predictions(path)
        pairs = [(r["raw_output"], (r.get("gt") or {}).get("result")) for r in recs]
        t = structured_consistency_table(pairs, specs, primary="score")
        print(f"\n[{cond}]  ({path})")
        _print_table(t)


def _enum_value(v):
    return v.value if v is not None else None


# ---------------------------------------------------------------------------
# 2) Function-calling benchmark (synthetic)
# ---------------------------------------------------------------------------


def _make_call(name: str, city: str) -> str:
    return (
        "I'll call the weather API for you: "
        + json.dumps(
            {"function": {"name": name, "arguments": json.dumps({"city": city})}}
        )
    )


FUNCTION_CALL_SAMPLES: list[tuple[str, tuple[str, str]]] = [
    # (raw_output, (gt_name, gt_city))
    (_make_call("get_weather", "SF"), ("get_weather", "SF")),      # A (coherent ✓)
    (_make_call("get_weather", "LA"), ("get_weather", "LA")),      # A
    (_make_call("get_weather", "LA"), ("get_weather", "SF")),      # both wrong in city -> primary=name rescues, city != GT
    (_make_call("weather_lookup", "SF"), ("get_weather", "SF")),   # name != GT, city = GT -> C_inv
    (_make_call("weather_lookup", "LA"), ("get_weather", "SF")),   # both wrong -> D or Fragmented
    ("The weather is cold.", ("get_weather", "SF")),               # U (no parse)
]


def demo_function_calling() -> None:
    _banner("2. Function calling (synthetic) — the same rescue pattern on JSON")
    specs = [
        FieldSpec("name", extract_function_name),
        # Primary field is `name` because benchmarks like Berkeley's function-calling
        # leaderboard often score on name-match first and arguments second.
        FieldSpec("city", lambda raw: extract_function_argument(raw, "city")),
    ]
    # GT for the generic machinery is a hashable; use a tuple-of-tuples only
    # for the classifier's per-field comparison to GT-of-each field. To make
    # this work we pass the *name* as GT and treat the city as the rescue channel.
    pairs: list[tuple[str, object]] = []
    for raw, (gt_name, _) in FUNCTION_CALL_SAMPLES:
        pairs.append((raw, gt_name))
    t = structured_consistency_table(pairs, specs, primary="name")
    print("  (GT: function name; rescue channel: city argument)")
    _print_table(t)


# ---------------------------------------------------------------------------
# 3) Structured JSON output with 3 fields
# ---------------------------------------------------------------------------


def _make_intent(intent: str, slot1: str, slot2: str) -> str:
    return json.dumps({"intent": intent, "slot1": slot1, "slot2": slot2})


INTENT_SAMPLES = [
    (_make_intent("book_flight", "SFO", "JFK"), "book_flight"),   # A
    (_make_intent("book_flight", "SFO", "JFK"), "book_hotel"),    # B (self-consistent wrong)
    (_make_intent("book_flight", "UNK", "UNK"), "book_flight"),   # C (intent matches, slots don't match intent? not a perfect illustration — but slots also happen to agree w/ each other)
    (_make_intent("book_hotel", "SFO", "JFK"), "book_flight"),    # D / C_inv depending on slot comparison
    ("not json", "book_flight"),                                  # U
]


def demo_three_field_json() -> None:
    _banner("3. Three-field JSON intent benchmark")
    specs = [
        FieldSpec("intent", lambda raw: extract_json_path(raw, "intent")),
        FieldSpec("slot1", lambda raw: extract_json_path(raw, "slot1")),
        FieldSpec("slot2", lambda raw: extract_json_path(raw, "slot2")),
    ]
    t = structured_consistency_table(INTENT_SAMPLES, specs, primary="intent")
    print("  (GT: intent; other fields: slot1, slot2)")
    _print_table(t)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    demo_football()
    demo_function_calling()
    demo_three_field_json()
    print("\nTakeaway: when primary_accuracy ≫ coherence_required_accuracy,")
    print("the benchmark is crediting the parser for what the model did not commit to.")


if __name__ == "__main__":
    main()
