# football-llm-scaling

**Is a fine-tuned model forecasting football results, or remembering them?**

[![tests](https://img.shields.io/badge/tests-87%20passing-brightgreen)](tests/)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Does fine-tuning beat prompting, or does the grading make it look that way? I
fine-tuned Llama 3.1 8B with QLoRA on World Cup matches and compared it with
5-shot prompting of the same model on the 64 matches of 2022. Each answer
states a result and a score. On named matches, if the score takes precedence the
192-match model leads by 21.9 points (exact McNemar p = 0.007); if the stated
result takes precedence it trails by 10.9; if both must match the truth the
difference is 1.6 points. The 32.8-point swing between the first two rules is
significant (p < 0.001).

The credited answers are recalled. The fine-tuned model writes the exact final
score of 42 of the 64 named matches, where no constant guess gets more than 7
and a Poisson model of the same statistics gets 5, and its stated result
contradicts that score in 17 of them. The 2022 World Cup predates the base
model's cutoff and is not in the training data. With the original anonymization,
which hides team names but keeps coaches and stadiums, the model still writes 20
real scores and names one of the real teams in 46 answers. With those hidden
too, it writes about 5% (see [Controls](#controls-october-2026)).

This is work in progress.

## Result

Each answer has three lines: `Prediction:` (the result), `Score:` and
`Reasoning:`. I score every answer three ways. The test set is the 64 matches
of the 2022 World Cup, each asked once with team names and once anonymized
(128 answers). Brackets are Wilson 95% intervals.

| Condition | Score-first: score overrides result | Stated result only | Coherent: result = score = truth | Predicts home win |
| --- | ---: | ---: | ---: | ---: |
| Always home win | 45.3 [37.0, 53.9] | | | 100% |
| ICL, 5-shot | 49.2 [40.7, 57.8] | 56.2 [47.6, 64.5] | 41.4 [33.2, 50.1] | 82% |
| CoT, 5-step | 44.5 [36.2, 53.2] | 53.1 [44.5, 61.6] | 35.2 [27.4, 43.8] | 99% |
| QLoRA, n = 48 | 46.1 [37.7, 54.7] | 44.5 [36.2, 53.2] | 44.5 [36.2, 53.2] ¹ | 98% |
| QLoRA, n = 96 | 51.6 [43.0, 60.0] | 32.0 [24.6, 40.5] | 28.1 [21.1, 36.5] | 21% |
| **QLoRA, n = 192** | **61.7** [53.1, 69.7] | 47.7 [39.2, 56.3] | 42.2 [34.0, 50.8] | 45% |
| QLoRA, n = 384 ² | 50.8 [42.2, 59.3] | 39.8 [31.8, 48.5] | 32.0 [24.6, 40.5] | 52% |

¹ n = 48 predicts a home win on 98% of inputs, so its fields agree trivially.
² n = 384 is the original adapter, trained on 192 named and 192 anonymized
rows. The smaller adapters saw named rows only.

The fine-tuned model's lead comes from answers that contradict themselves. On
the named half, which is in-distribution for every adapter, I took every answer
whose stated result and score disagree and asked which one was right. For the
fine-tuned models the score was right 28 times to 2 at n = 96, 18 to 2 at
n = 192 and 12 to 2 at n = 384. For the prompted model it was the reverse:
4 to 9 for ICL and 6 to 13 for CoT. So the score-first rule adds points to
fine-tuning and takes them from prompting. Of the 18 answers it rescues for
n = 192, 17 carry the exact real score.

Fine-tuning learned the format. Every adapter answered in three lines on all
128 inputs (the 48-match adapter wrote its labels in title case). It did not learn to make the result and the score agree: on the
named half, the two agree on 96.9% of answers at n = 48, 46.9% at n = 96 and
64.1% at n = 192. Any benchmark that scores several fields of one answer
(function calls, JSON, tool use) can hide the same failure. Reporting
coherence-required accuracy next to the usual number catches it.

![Score-based accuracy, stated-result accuracy and field agreement for every condition](results/figures/consistency_curve.png)

## Controls (October 2026)

Memory is a known risk when language models forecast past events
([Gao et al., 2025](https://arxiv.org/abs/2512.23847);
[Gorla & Puduppully, 2026](https://arxiv.org/abs/2602.04031)). These controls
test it for this fine-tuned model on the same matches. Unless stated, they use
the 64 named 2022 matches and three training seeds. The lead is the fine-tuned
model's accuracy under the score-first rule minus 5-shot prompting's, in points,
with a 95% bootstrap interval over matches.

| Condition | Exact real score | Lead over prompting |
| --- | ---: | ---: |
| Real scores (the n = 192 setup above) | 62.0% | +19.3 [6.8, 31.8] |
| Real scores, ten seeds | 63.4% | +20.0 [7.5, 32.8], p = 0.003 |
| Fake scores: same matches and stated results, plausible fake scores | 9.9% | −5.7 [−17.7, 5.7] |
| Full masking: team names, coaches, stadium, round and World Cup record hidden | 5.2% | −9.4 [−26.6, 7.8] |
| 2026 World Cup: 104 matches after the model's training data ends | 8.7% | +2.2 [−9.0, 12.8] |

- The best constant guess writes the exact score of 10.9% of 2022 matches and
  11.5% of 2026 matches.
- Mistral 7B and Qwen2.5 14B also recall the real scores after fine-tuning
  (44.8% and 57.3%), and fake scores remove that recall (9.9% and 8.3%). Their
  lead over prompting was small to begin with (+3.1 and +6.2 points; both
  intervals include zero).
- A simple Elo rating, which never predicts a draw, picks the winner of 54.7% of
  2022 matches and 64.4% of 2026 matches. Prompting's stated result gets 56.2%
  and 61.5%.

The control runs are not yet in this repository.

## Setup

- **Task.** From two teams' aggregated per-player statistics (API-Football,
  prior seasons only), predict the result and the score.
- **Data.** Training: the 192 matches of the 2010, 2014 and 2018 World Cups.
  Test: the 64 matches of 2022. Each match appears once with team names and
  once anonymized. Both sets are in [`data/`](data/).
- **Fine-tuning.** QLoRA on Llama-3.1-8B-Instruct (4-bit NF4, rank 16,
  α = 32, 3 epochs, learning rate 2e-4) at 48, 96 and 192 named matches.
  The subsets are nested and keep the class balance. n = 384 is the published
  adapter [`zanwenfu/football-llm-qlora`](https://huggingface.co/zanwenfu/football-llm-qlora).
- **Prompting.** The same base model. ICL uses 5 demonstrations (2 home wins,
  2 away wins, 1 draw); CoT uses a 5-step scaffold with 1,024 new tokens.
- **Decoding.** Temperature 0.1, top-p 0.9, repetition penalty 1.1. One seed,
  one T4.
- **Scoring.** The parser, the three views and the regime taxonomy are in
  [`docs/METHODS.md`](docs/METHODS.md). Paired comparisons use an exact
  McNemar test over the same 128 inputs.

## Limitations

- One task and one prompt format. The table above uses one 8B model and one
  seed per budget; the controls add seeds and two more models. The n = 96 drop
  in field agreement needs more seeds before it is a trend.
- The 128 answers are 64 matches asked twice, so the intervals and p-values
  above, which treat them as independent, are optimistic. On either half alone (64 answers)
  they are about ±12 points.
- n = 384 changes the training mix as well as the size, so I make no scaling
  claim from n = 192 to n = 384.
- The function-calling and JSON examples in
  [`scripts/demo_structured_rescue.py`](scripts/demo_structured_rescue.py) are
  synthetic. They show how to apply the check; they are not evidence that the
  gap appears on those benchmarks.

## Reproduce

Every table and figure regenerates from the saved predictions in
[`results/raw/`](results/raw/) in about a second, with no GPU:

```bash
pip install -e .
make all
```

To retrain and re-evaluate (GPU, access to Llama 3.1 8B):

```bash
pip install -e '.[train]'
bash scripts/reproduce_paper.sh --train --eval
```

[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) covers each step, the
software versions, seeds and multi-seed runs.

**Data.** The prompts in [`data/`](data/) are built from public match
records (team and player statistics from earlier matches, and final scores),
retrieved through [API-Football](https://www.api-football.com).

## Use the check on your own outputs

```python
from data import load_predictions
from metrics import compute_all_splits

m = compute_all_splits(load_predictions("results/raw/scaling_predictions_n192.json"))["overall"]
print(f"{m.score_acc:.3f} {m.text_acc:.3f} {m.coherence_acc:.3f}")  # 0.617 0.477 0.422
```

[`src/structured.py`](src/structured.py) applies the same check to any set of
fields, such as a function name and its arguments. It reports primary accuracy
and coherence-required accuracy side by side; a large gap between the two is
the signature.

## Related work

The closest work is on how answer extraction moves scores:
[xFinder](https://arxiv.org/abs/2405.11874) (Yu et al., ICLR 2025),
[Lessons from the Trenches](https://arxiv.org/abs/2405.14782) (Biderman et al.,
2024) and [Finding Answers in Thought Matters](https://arxiv.org/abs/2510.14773)
(Jo et al., 2025). They study how extraction rules change measured accuracy;
this study shows a rule that lets one field override another changing which method wins.
Related threads are the mismatch between how a model is scored and what it
generates ([Lyu et al., 2024](https://arxiv.org/abs/2402.13887)), reasoning
that does not match the answer ([Turpin et al., NeurIPS 2023](https://arxiv.org/abs/2305.04388)),
format learned from few examples ([LIMA](https://arxiv.org/abs/2305.11206)),
controlled fine-tuning versus ICL comparisons
([Mosbach et al., Findings of ACL 2023](https://arxiv.org/abs/2305.16938)) and
format restrictions ([Tam et al., EMNLP 2024 Industry](https://arxiv.org/abs/2408.02442)).
Requiring every field to be right is standard elsewhere, for example joint goal
accuracy in dialogue state tracking ([MultiWOZ](https://arxiv.org/abs/1810.00278)).

## Repository

| Path | Contents |
| --- | --- |
| `src/` | Parser, metrics, regime taxonomy, data loading, training and evaluation |
| `scripts/` | Train, evaluate, analyze, plot, and the structured-output demo |
| `results/` | Saved predictions (`raw/`), tables, figures and example contradictions |
| `data/` | Training and test prompts |
| `notebooks/` | The original research notebooks |
| `docs/` | [Methods](docs/METHODS.md), [figures](docs/FIGURES.md), [reproducibility](docs/REPRODUCIBILITY.md) |
| `tests/` | Unit tests, run in CI |

## Citing

Please cite the repository:

```bibtex
@misc{fu2026parserrescue,
  author       = {Fu, Zanwen},
  title        = {football-llm-scaling: Is a Fine-Tuned Model Forecasting
                  Football Results, or Remembering Them?},
  year         = {2026},
  howpublished = {\url{https://github.com/zanwenfu/football-llm-scaling}}
}
```

[CITATION.cff](CITATION.cff) has machine-readable citation metadata.

## License

MIT. See [LICENSE](LICENSE).
