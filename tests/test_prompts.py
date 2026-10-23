"""The prompts the code builds are the prompts the saved predictions used."""

from __future__ import annotations

import json
from pathlib import Path

from config import DEFAULT_COT_GEN, DEFAULT_GEN, DEFAULT_ICL_GEN, EVAL_DATA, TRAIN_DATA
from prompts import SYSTEM_PROMPT_BASE, SYSTEM_PROMPT_COT, build_qlora_messages

REPO = Path(__file__).resolve().parents[1]


def _first_system(path: str) -> str:
    with (REPO / path).open(encoding="utf-8") as f:
        return json.loads(f.readline())["messages"][0]["content"]


def test_base_system_prompt_is_the_one_in_the_data():
    assert SYSTEM_PROMPT_BASE == _first_system(TRAIN_DATA) == _first_system(EVAL_DATA)


def test_cot_system_prompt_extends_the_base_one():
    assert SYSTEM_PROMPT_COT.startswith(SYSTEM_PROMPT_BASE)
    assert "Step 5" in SYSTEM_PROMPT_COT


def test_qlora_messages_match_the_training_rows():
    with (REPO / TRAIN_DATA).open(encoding="utf-8") as f:
        row = json.loads(f.readline())
    assert build_qlora_messages(row["messages"][1]["content"]) == row["messages"][:2]


def test_token_budgets_match_the_saved_runs():
    assert (DEFAULT_GEN.max_new_tokens, DEFAULT_ICL_GEN.max_new_tokens, DEFAULT_COT_GEN.max_new_tokens) == (400, 300, 1024)
