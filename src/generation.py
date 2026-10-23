"""Model loading and text generation — the only module that needs a GPU.

Lazy-imports ``torch``, ``transformers``, and ``peft`` so that analysis code
(metrics, consistency, plotting) can be imported on CPU-only machines without
pulling in 3+ GB of ML dependencies.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from config import (
    BASE_MODEL_ID,
    DEFAULT_GEN,
    DEFAULT_QUANT,
    GenerationConfig,
    QuantizationConfig,
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase  # noqa: F401


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_base_model(
    model_id: str = BASE_MODEL_ID,
    quant_cfg: QuantizationConfig = DEFAULT_QUANT,
    device_map: str = "auto",
):
    """Load the 4-bit-quantized base model and its tokenizer."""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    bnb = BitsAndBytesConfig(
        load_in_4bit=quant_cfg.load_in_4bit,
        bnb_4bit_quant_type=quant_cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=quant_cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg.bnb_4bit_compute_dtype),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Llama-3.1 ships a reserved pad id for SFT right-padding
    pad_id = tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")
    if pad_id is not None and pad_id != tokenizer.unk_token_id:
        tokenizer.pad_token_id = pad_id
    elif tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map=device_map,
    )
    return model, tokenizer


def attach_adapter(base_model, adapter_id_or_path: str):
    """Wrap ``base_model`` with a PEFT adapter loaded from the Hub or disk."""
    from peft import PeftModel

    return PeftModel.from_pretrained(base_model, adapter_id_or_path)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    cfg: GenerationConfig = DEFAULT_GEN,
) -> str:
    """Apply the chat template and generate a response string."""
    import torch

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    torch.manual_seed(cfg.seed)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            repetition_penalty=cfg.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Batch eval loop (shared by QLoRA / ICL / CoT)
# ---------------------------------------------------------------------------


def run_eval_loop(
    eval_samples: Iterable[dict[str, Any]],
    model,
    tokenizer,
    build_messages,
    cfg: GenerationConfig = DEFAULT_GEN,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Generate a response for each eval sample and collect a result record.

    ``build_messages(sample) -> list[dict]`` lets the caller plug in the
    QLoRA / ICL / CoT message builder without this module knowing about
    prompt construction.

    The returned records match the schema consumed by :mod:`metrics` and
    :mod:`consistency`::

        {"sample_idx", "is_anon", "gt", "raw_output", "output_len", "pred"}

    ``pred`` is populated using :func:`parsing.parse_output` and the legacy
    score-overrides-text resolution, matching the shape in
    ``results/raw/*.json``.
    """
    from parsing import (
        parse_output,
        resolve_score_overrides_text,
    )

    records: list[dict[str, Any]] = []
    for idx, sample in enumerate(eval_samples):
        messages = build_messages(sample)
        raw = generate(model, tokenizer, messages, cfg)
        p = parse_output(raw)
        effective = resolve_score_overrides_text(p)
        records.append(
            {
                "sample_idx": idx,
                "is_anon": bool(sample.get("is_anon") or sample.get("metadata", {}).get("is_anon")),
                "gt": sample.get("gt") or _gt_from_metadata(sample),
                "pred": {
                    "result": effective.value if effective else None,
                    "home_goals": p.home_goals,
                    "away_goals": p.away_goals,
                    "parsed": effective is not None,
                },
                "raw_output": raw,
                "output_len": len(raw),
            }
        )
        if verbose and (idx + 1) % 16 == 0:
            print(f"[eval] {idx + 1} / ? complete")
    return records


def _gt_from_metadata(sample: dict[str, Any]) -> dict[str, Any]:
    """Back-compat helper for eval samples that store GT in ``metadata``."""
    meta = sample.get("metadata") or {}
    return {
        "result": meta.get("result"),
        "home_goals": meta.get("home_goals"),
        "away_goals": meta.get("away_goals"),
        "parsed": True,
    }
