"""End-to-end evaluation orchestrator: QLoRA / ICL / CoT dispatch.

The three conditions differ only in:

1. Whether a LoRA adapter is attached to the base model (QLoRA only).
2. Which :mod:`prompts` builder is used.
3. Which :class:`GenerationConfig` is used (CoT needs 1024 new tokens).

This module exposes a single :func:`evaluate_condition` that returns the same
result schema regardless of condition, so analysis code downstream never needs
to branch on which method produced a prediction dump.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Literal, Optional

from config import (
    BASE_MODEL_ID,
    DEFAULT_COT_GEN,
    DEFAULT_GEN,
    DEFAULT_ICL,
    GenerationConfig,
    ICLConfig,
)
from data import save_predictions

Condition = Literal["qlora", "icl", "cot"]


def evaluate_condition(
    condition: Condition,
    eval_samples: list[dict[str, Any]],
    train_samples: Optional[list[dict[str, Any]]] = None,
    adapter_id_or_path: Optional[str] = None,
    gen_cfg: Optional[GenerationConfig] = None,
    icl_cfg: ICLConfig = DEFAULT_ICL,
    model_id: str = BASE_MODEL_ID,
    save_to: Optional[str | Path] = None,
) -> list[dict[str, Any]]:
    """Run one eval condition end-to-end and optionally persist predictions.

    Parameters
    ----------
    condition
        ``"qlora"`` | ``"icl"`` | ``"cot"``.
    eval_samples
        Eval records. Each must have ``metadata`` or ``gt`` with the ground-
        truth result + goals, an ``is_anon`` flag, and either a ``messages``
        list whose last user turn is the stats prompt, or a ``user_prompt``
        string.
    train_samples
        Required only for ``condition="icl"`` (demo selection).
    adapter_id_or_path
        Required only for ``condition="qlora"``.
    save_to
        If given, write the result list to this JSON path.
    """
    from generation import (  # noqa: PLC0415
        attach_adapter,
        load_base_model,
        run_eval_loop,
    )
    from prompts import (  # noqa: PLC0415
        build_cot_messages,
        build_icl_messages,
        build_qlora_messages,
        select_stratified_demos,
    )

    if gen_cfg is None:
        gen_cfg = DEFAULT_COT_GEN if condition == "cot" else DEFAULT_GEN

    model, tokenizer = load_base_model(model_id=model_id)
    if condition == "qlora":
        if adapter_id_or_path is None:
            raise ValueError("condition='qlora' requires adapter_id_or_path")
        model = attach_adapter(model, adapter_id_or_path)

    if condition == "icl":
        if train_samples is None:
            raise ValueError("condition='icl' requires train_samples for demo selection")
        demo_idx = select_stratified_demos(train_samples, icl_cfg)
        demos_named, demos_anon = _materialize_demos(train_samples, demo_idx)

        def build(sample: dict[str, Any]) -> list[dict[str, str]]:
            user = _user_prompt(sample)
            demos = demos_anon if sample.get("is_anon") else demos_named
            return build_icl_messages(demos, user)

    elif condition == "cot":

        def build(sample: dict[str, Any]) -> list[dict[str, str]]:
            return build_cot_messages(_user_prompt(sample))

    else:  # qlora

        def build(sample: dict[str, Any]) -> list[dict[str, str]]:
            return build_qlora_messages(_user_prompt(sample))

    records = run_eval_loop(eval_samples, model, tokenizer, build, gen_cfg)

    if save_to is not None:
        save_predictions(records, save_to)
    return records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user_prompt(sample: dict[str, Any]) -> str:
    """Extract the user prompt from a HuggingFace chat-format sample."""
    if "user_prompt" in sample:
        return sample["user_prompt"]
    for msg in sample.get("messages", []):
        if msg.get("role") == "user":
            return msg["content"]
    raise KeyError("sample has no 'user_prompt' or user-role message")


def _materialize_demos(
    train_samples: list[dict[str, Any]],
    demo_idx: Iterable[int],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Extract (user, assistant) pairs and build named + anon versions.

    For each selected demo we return the named version and a best-effort
    anonymized version (matched by result+score in the same train split, with
    the "Team A"/"Team B" anonymization expected to already be encoded in the
    training data under the ``is_anon`` flag).
    """
    named_pool: dict[int, tuple[str, str]] = {}
    anon_pool: dict[tuple[str, int, int], tuple[str, str]] = {}
    for i, rec in enumerate(train_samples):
        msgs = rec.get("messages", [])
        user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        if rec.get("is_anon") or (rec.get("metadata") or {}).get("is_anon"):
            key = (
                (rec.get("metadata") or {}).get("result"),
                (rec.get("metadata") or {}).get("home_goals"),
                (rec.get("metadata") or {}).get("away_goals"),
            )
            anon_pool[key] = (user, assistant)
        else:
            named_pool[i] = (user, assistant)

    named_demos: list[tuple[str, str]] = []
    anon_demos: list[tuple[str, str]] = []
    for idx in demo_idx:
        named = named_pool.get(idx)
        if named is None:
            continue
        named_demos.append(named)
        meta = (train_samples[idx].get("metadata") or {})
        key = (meta.get("result"), meta.get("home_goals"), meta.get("away_goals"))
        anon_demos.append(anon_pool.get(key, named))
    return named_demos, anon_demos
