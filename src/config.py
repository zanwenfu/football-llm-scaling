"""Hyperparameters and constants.

Everything that might look like a magic number elsewhere in the package is
defined here. Each config is a frozen dataclass so it prints cleanly and can be
hashed into a run identifier. Values match the paper and the original Colab
training notebook; see METHODS in the paper for the rationale.
"""

from __future__ import annotations

from dataclasses import dataclass

# Model identifiers ----------------------------------------------------------

BASE_MODEL_ID: str = "meta-llama/Llama-3.1-8B-Instruct"
"""HuggingFace ID of the base model fine-tuned and prompted in all experiments."""

ADAPTER_HUB_ID: str = "zanwenfu/football-llm-qlora"
"""The pre-existing `n=384` adapter reused from the base football-llm repo.

Trained on 192 named + 192 anonymized examples. The `n<=192` adapters in the
scaling ablation are trained locally by `scripts/01_train_scaling.py`.
"""

TRAIN_DATASET_HUB_ID: str = "zanwenfu/football-llm-train"
EVAL_DATASET_HUB_ID: str = "zanwenfu/football-llm-eval"

# Evaluation / dataset sizes -------------------------------------------------

TRAIN_TOTAL_NAMED: int = 192
"""2010 + 2014 + 2018 World Cup matches (named split)."""

EVAL_TOTAL: int = 128
"""2022 World Cup, 64 named + 64 anonymized."""

SCALING_BUDGETS: tuple[int, ...] = (48, 96, 192, 384)
"""Training-set sizes swept in the scaling ablation.

48, 96, 192 are nested stratified prefixes of the named training split
(verified `s_48 ⊂ s_96 ⊂ s_192`). n=384 reuses the pre-existing
`zanwenfu/football-llm-qlora` adapter trained on 192 named + 192 anonymized.
The n=384 configuration is therefore not directly comparable to the smaller
adapters as a "scale" data point — see the Limitations section of the paper.
"""

# Class priors ---------------------------------------------------------------

BASELINE_HOME_WIN_RATE: float = 0.453125
"""Fraction of home_win outcomes on the 2022 eval split (58/128).

Equal to the always-home-win accuracy baseline.
"""

WC_HOME_WIN_PRIOR: float = 0.40
WC_AWAY_WIN_PRIOR: float = 0.35
WC_DRAW_PRIOR: float = 0.25
"""Class distribution used by the random-weighted baseline."""

# Labels ---------------------------------------------------------------------

LABELS: tuple[str, ...] = ("home_win", "away_win", "draw")

# Statistics -----------------------------------------------------------------

WILSON_Z_95: float = 1.959963984540054
"""Two-sided 95% critical value for the Wilson score interval."""


@dataclass(frozen=True)
class QLoRAConfig:
    """4-bit NF4 QLoRA setup targeting the seven standard Llama projections."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


@dataclass(frozen=True)
class QuantizationConfig:
    """BitsAndBytes 4-bit quantization matching the paper setup."""

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "float16"


@dataclass(frozen=True)
class TrainingConfig:
    """SFT training hyperparameters (effective batch 16, 3 epochs, cosine)."""

    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.10
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    max_seq_length: int = 768
    seed: int = 42
    logging_steps: int = 5
    save_strategy: str = "epoch"
    bf16: bool = False
    fp16: bool = True


@dataclass(frozen=True)
class GenerationConfig:
    """Decoding config for every evaluation condition.

    Kept identical across QLoRA / ICL / CoT to isolate the effect of training
    data from sampling randomness. `temperature=0.1` is low enough to make runs
    effectively deterministic at fixed seed while still leaving the parser
    stress-tested on the rare token-level variation.
    """

    temperature: float = 0.1
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True
    max_new_tokens: int = 400
    seed: int = 42


@dataclass(frozen=True)
class CoTGenerationConfig(GenerationConfig):
    """CoT needs a larger budget to reach the final `Prediction:` line.

    The paper reran CoT (`notebooks/03b_cot_rerun.ipynb`) after discovering the
    original 300-token cap was truncating before the conclusion.
    """

    max_new_tokens: int = 1024


@dataclass(frozen=True)
class EvalConfig:
    """How evaluation splits are constructed and stored."""

    n_named: int = 64
    n_anon: int = 64
    n_eval_total: int = 128
    seed: int = 42


@dataclass(frozen=True)
class ICLConfig:
    """5-shot stratified demo selection (2 home / 2 away / 1 draw)."""

    n_demos: int = 5
    n_home: int = 2
    n_away: int = 2
    n_draw: int = 1
    seed: int = 42


# Convenience -------------------------------------------------------------

DEFAULT_QLORA = QLoRAConfig()
DEFAULT_QUANT = QuantizationConfig()
DEFAULT_TRAIN = TrainingConfig()
DEFAULT_GEN = GenerationConfig()
DEFAULT_COT_GEN = CoTGenerationConfig()
DEFAULT_EVAL = EvalConfig()
DEFAULT_ICL = ICLConfig()
