"""System / user / demo messages for QLoRA, ICL, and CoT conditions.

Each condition uses the same base system prompt and user-prompt template; the
variations are:

* **QLoRA**: system + user only (no demos, no scaffold).
* **ICL**: system + 5 stratified demo turns + user.
* **CoT**: system + reasoning scaffold + user.

Demos are sampled stratified 2 home / 2 away / 1 draw with ``seed=42`` from the
*named* training split. When the eval sample is anonymized, anonymized
versions of the same demonstrations are used.
"""

from __future__ import annotations

import random
from typing import Any

from config import ICLConfig

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

# The system message stored with every training and test row, and the one the
# saved predictions were generated with (notebooks/01-04).
SYSTEM_PROMPT_BASE: str = (
    "You are a football match prediction model. Given team stats, predict the "
    "result, score, and brief reasoning."
)

# notebooks/03b_cot_rerun.ipynb appends this scaffold to the base system message.
COT_SCAFFOLD: str = (
    "\n\nBefore your final answer, reason step-by-step:\n"
    "Step 1: Compare the two teams' squad goal output. Which is higher and by how much?\n"
    "Step 2: Compare the quality of each team's top scorer.\n"
    "Step 3: Compare defensive stats (tackles, clean sheets, defensive rating).\n"
    "Step 4: Identify any conflicts between these signals and weigh them.\n"
    "Step 5: Based on the above, predict the outcome.\n\n"
    "After your reasoning, conclude in EXACTLY this format on separate lines:\n"
    "Prediction: <home_win|away_win|draw>\n"
    "Score: <home_goals>-<away_goals>\n"
    "Reasoning: <one sentence summary>\n"
    "Keep your step-by-step reasoning concise (2-3 sentences per step) so you have "
    "room for the conclusion."
)

SYSTEM_PROMPT_COT: str = SYSTEM_PROMPT_BASE + COT_SCAFFOLD

# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------


def build_qlora_messages(user_prompt: str) -> list[dict[str, str]]:
    """System + user turn used at QLoRA training and eval time."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_BASE},
        {"role": "user", "content": user_prompt},
    ]


def build_icl_messages(
    demos: list[tuple[str, str]],
    user_prompt: str,
) -> list[dict[str, str]]:
    """System + N demo (user, assistant) turns + final user query.

    ``demos`` is a list of (user_prompt, assistant_response) pairs drawn from
    the training split by :func:`select_stratified_demos`.
    """
    msgs: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT_BASE}]
    for demo_user, demo_assistant in demos:
        msgs.append({"role": "user", "content": demo_user})
        msgs.append({"role": "assistant", "content": demo_assistant})
    msgs.append({"role": "user", "content": user_prompt})
    return msgs


def build_cot_messages(user_prompt: str) -> list[dict[str, str]]:
    """System (with CoT scaffold) + user turn."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_COT},
        {"role": "user", "content": user_prompt},
    ]


# ---------------------------------------------------------------------------
# Stratified demo selection
# ---------------------------------------------------------------------------


def select_stratified_demos(
    train_records: list[dict[str, Any]],
    cfg: ICLConfig | None = None,
) -> list[int]:
    """Pick 2 home + 2 away + 1 draw indices from the named training split.

    Returns a list of indices into ``train_records``. Deterministic given
    ``cfg.seed``. The corresponding (user, assistant) pairs are retrieved by
    the caller because their exact location depends on the training schema
    (``messages[1].content`` and ``messages[2].content`` for the HuggingFace
    chat-format training data).
    """
    if cfg is None:
        cfg = ICLConfig()
    rng = random.Random(cfg.seed)
    by_label: dict[str, list[int]] = {"home_win": [], "away_win": [], "draw": []}
    for idx, rec in enumerate(train_records):
        label = _gt_label(rec)
        if label in by_label:
            by_label[label].append(idx)

    picks: list[int] = []
    picks += rng.sample(by_label["home_win"], cfg.n_home)
    picks += rng.sample(by_label["away_win"], cfg.n_away)
    picks += rng.sample(by_label["draw"], cfg.n_draw)
    rng.shuffle(picks)
    return picks


def _gt_label(record: dict[str, Any]) -> str | None:
    meta = record.get("metadata") or record.get("gt") or {}
    return meta.get("result")
