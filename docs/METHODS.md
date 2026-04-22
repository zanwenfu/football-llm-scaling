# Methods

Technical notes on the parts of the implementation where the "right" choice
isn't obvious from reading the code. For experimental results and discussion
see [paper/ECE590_Final_Project_Report.pdf](../paper/ECE590_Final_Project_Report.pdf).

## 1. The three output channels

Every model output in the evaluation contains (up to) three fields:

```
Prediction: <home_win | away_win | draw>
Score:      <h>-<a>
Reasoning:  <sentence>
```

The standard convention baked into the original `eval_harness.ipynb` computes
a *single* result label per sample using the rule "if the `Score:` line
parses to a realistic score, use its implied winner; otherwise fall back to
the `Prediction:` line." This rule was originally motivated by the
observation that the text label sometimes drops out of CoT reasoning, while
the score is almost always present.

`src/parsing.py` splits this into three independent
extractors so that the analysis code can measure the cost of the rule
instead of silently absorbing it:

* `extract_text_label(raw)` returns only what the `Prediction:` line says.
* `extract_score_label(raw)` returns only what the `Score:` line implies.
* `extract_reasoning_label(raw)` is a noisy keyword heuristic over the
  `Reasoning:` paragraph, documented as such and only used for appendix
  analysis.
* `resolve_score_overrides_text(parsed)` re-applies the legacy rule so code
  that wants to reproduce the original `eval_harness` numbers can.

## 2. The hardened score parser

The original evaluation contained a bare-digit fallback that matched any
`\d+-\d+` anywhere in the output. On CoT traces this routinely matched
aggregate statistics like `"squad goals 303-167"` in the reasoning paragraph,
producing obviously wrong scores like `(303, 167)`. The hardened parser
([`parsing.extract_score`](../src/parsing.py)) does
three things:

1. Prefers the labeled `Score:` form anywhere in the output.
2. Only falls back to the bare `\d+-\d+` form in the last 400 characters
   (where a concluding score would appear).
3. Requires both sides of the score to be in `[0, 15]`.

The false-positive case is a named test:
`tests/test_parsing.py::TestExtractScore::test_squad_goal_totals_are_not_mis_parsed`.

## 3. Three accuracy views

`metrics.compute_metrics` returns three different accuracy numbers jointly:

| attr             | meaning                                                                 |
| ---------------- | ----------------------------------------------------------------------- |
| `score_acc`      | Legacy "score overrides text" — the headline column in the paper.       |
| `text_acc`       | What the `Prediction:` line actually says — ignores the score.          |
| `coherence_acc`  | Counts a prediction correct only if text == score == GT (Regime A).     |

The spread between these three for a single condition is the diagnostic
signal the paper is about. If score_acc ≫ coherence_acc for a condition,
that condition is benefiting from the parser resolving its own
contradictions.

All three come with Wilson 95% two-sided CIs (half-width in pp; `wilson_ci`
and `wilson_halfwidth_pp`).

## 4. Regime taxonomy

`consistency.Regime` enumerates six mutually exclusive per-sample classes:

| regime | text_label | score_label | ground_truth | name                    |
| :----: | :--------: | :---------: | :----------: | ----------------------- |
|   A    |     X      |      X      |      X       | coherent success        |
|   B    |     X      |      X      |      Y       | self-consistent mistake |
|   C    |     X      |      Y      |      Y       | parser-rescued          |
|   C'   |     X      |      Y      |      X       | parser-penalized        |
|   D    |     X      |      Y      |      Z       | fragmented              |
|   U    |   either   |   either    |    either    | unparseable             |

Regime C is the one the paper's title is about. The coherence-required
accuracy in §3 is exactly `regime_counts.A / regime_counts.total`.

## 5. Disagreement directionality

`consistency.disagreement_directionality(records)` returns, among
*text≠score* disagreements only: how many have score matching GT (C), how
many have text matching GT (C'), and how many have neither (D). The ratio
C/C' answers the question "which channel is more reliable on this
condition's disagreements?". For ICL/CoT the ratio is < 1 (text is more
reliable), for QLoRA at n ≥ 96 it flips to 6–14× (score is nominally more
reliable, because the model has learned to hedge with realistic scores
even when the text label is wrong).

Computed on the **named split only** in the paper to avoid mixing
in-distribution and OOD samples. The script
[`scripts/03_analyze_results.py`](../scripts/03_analyze_results.py) matches
this choice.

## 6. Nested stratified prefixes for the scaling ablation

The scaling ablation trains QLoRA adapters at n ∈ {48, 96, 192}. If the three
training sets were independently drawn, the trend line would confound data
budget with sample identity. `data.stratified_nested_prefix_indices`
enforces `s_48 ⊂ s_96 ⊂ s_192` by building a single per-class shuffled index
list and taking class-proportional prefixes from it at each budget. The
function verifies the subset invariant before returning and raises if class
rounding off-by-ones somehow break it.

The n=384 configuration reuses the public `zanwenfu/football-llm-qlora`
adapter trained on 192 named + 192 anonymized — a different training-data
composition. The paper is careful not to treat n=192 → n=384 as a clean
scale data point (see §5 Limitations), and the package follows suit.

## 7. Generation config

All conditions share the same decoding config (`config.GenerationConfig`):
`temperature=0.1`, `top_p=0.9`, `repetition_penalty=1.1`. Low temperature
plus fixed seed makes runs effectively deterministic; the tiny remaining
token-level variance stress-tests the parser in the same way across
conditions. CoT uses `max_new_tokens=1024` because a 5-step scaffold does
not fit into 300 tokens while still reaching the concluding `Prediction:`
line (the original CoT run truncated before the conclusion; see
`notebooks/03b_cot_rerun.ipynb`).

## 8. Multi-seed aggregation (post-paper follow-up)

The paper's §5 Limitations flags that a single seed per budget cannot
distinguish a real effect from seed variance. [`aggregate.py`](../src/aggregate.py)
adds:

* `aggregate_across_seeds([records_seed_a, records_seed_b, ...])` — runs
  `compute_metrics` on each seed and returns an `AggregatedCondition` with
  mean, std, and percentile-bootstrap 95% CI for each of the three
  accuracy views. Bootstrap rather than Wilson because we want a CI on the
  *mean across seeds*, not on a single binomial proportion.
* `scripts/01_train_scaling.py --seeds 42 43 44 ...` trains a fresh
  adapter per seed, using a separate `--prefix-seed` for the sampler so
  the same data subset is used across seeds at each budget (isolating
  optimization stochasticity from data-subset stochasticity).
* `scripts/03_analyze_results.py` auto-detects the `_s{seed}` filename
  suffix, groups by budget, and writes `aggregated_metrics.json`.
* `scripts/04_make_figures.py` switches the scaling curve to error bars
  when more than one seed is present.

The bootstrap half-width is not a substitute for a principled SE on
k seeds — with k=3 it is still a rough estimate — but it is honest about
what the data supports and gives a visually comparable error bar
alongside the paper's Wilson intervals.

## 9. Generalized taxonomy for any multi-field benchmark (post-paper follow-up)

[`structured.py`](../src/structured.py) lifts the A / B / C / C' / D
taxonomy off football. The paper's final paragraph argues the parser-rescue
gap is likely to surface on function-calling, JSON-output, and tool-use
benchmarks — this module lets a reviewer drop the taxonomy into those
settings in a few lines.

* `FieldSpec(name, extract)` — describes one parseable channel.
* `classify_multifield({field: value_or_None}, ground_truth, primary)` —
  generalized classifier. For N=2 reduces exactly to the football case;
  for N≥3 the definitions are:

  * A — every parsed field equals the others AND the primary equals GT.
  * B — fields equal each other, primary ≠ GT.
  * C — fields disagree, primary = GT (*parser rescues the primary metric*).
  * C' — fields disagree, primary ≠ GT but some other field = GT.
  * D — fields disagree, nothing matches GT.
  * U — at least one field failed to parse.

* `structured_consistency_table(pairs, specs, primary)` — aggregates over
  a whole benchmark and returns `primary_accuracy`,
  `coherence_required_accuracy`, `primary_rescue_rate`, and
  `primary_penalty_rate`. A large gap between the first two is the
  benchmark's parser-rescue signature.

JSON / function-calling convenience helpers — `extract_json_block`,
`extract_json_path`, `extract_function_name`, `extract_function_argument` —
handle the common shapes. `scripts/demo_structured_rescue.py` runs three
worked examples: the paper's own football data through the generic
machinery (sanity check against the paper's numbers), an OpenAI-style
function-calling benchmark, and a three-field JSON intent benchmark.

## 10. What the package deliberately does not do

* **Live scoring against a real sportsbook or a production API.** The base
  repo has a `serving/` module for that; the scaling extension is evaluation
  and analysis only.
* **Multi-seed sweeps in the *paper's* results.** The paper runs a single seed per budget and calls
  this out as a limitation. The repo now *supports* multi-seed sweeps via
  `scripts/01_train_scaling.py --seeds ...`, but the paper's reported
  numbers are still single-seed. Re-running the sweep with 3+ seeds is
  the natural next experiment.
* **Attention-pattern analysis.** The original proposal pre-registered a
  layer-wise attention study. The paper explains why this was dropped:
  when the three output fields are produced by visibly different
  field-specific generation processes, a single-forward-pass attention
  analysis would not faithfully describe "what the model attends to when
  predicting."
