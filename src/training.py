"""QLoRA fine-tuning driver parameterized by data budget.

This is the one place the scaling ablation differs from the base football-llm
training recipe: we sweep ``n`` over :data:`config.SCALING_BUDGETS` using
nested stratified prefixes (see :mod:`data`). Every other hyperparameter is
held constant so the n-to-n differences are interpretable as data-budget
effects modulo seed variance.

Like :mod:`generation`, this module lazy-imports torch / transformers / peft /
trl so the analysis-only code path does not pay the import cost.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from config import (
    BASE_MODEL_ID,
    DEFAULT_QLORA,
    DEFAULT_QUANT,
    DEFAULT_TRAIN,
    QLoRAConfig,
    QuantizationConfig,
    TrainingConfig,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase  # noqa: F401


def train_qlora(
    train_records: list[dict[str, Any]],
    output_dir: str | Path,
    qlora_cfg: QLoRAConfig = DEFAULT_QLORA,
    quant_cfg: QuantizationConfig = DEFAULT_QUANT,
    train_cfg: TrainingConfig = DEFAULT_TRAIN,
    model_id: str = BASE_MODEL_ID,
    resume_if_exists: bool = True,
) -> Path:
    """Fine-tune a 4-bit Llama-3.1 base with a LoRA adapter on ``train_records``.

    ``train_records`` is a list of ``{"messages": [...]}`` dicts in the
    HuggingFace chat format, i.e. what the ``zanwenfu/football-llm-train``
    dataset yields. Returns the path to the saved adapter directory.

    If ``resume_if_exists`` is True and the output directory already contains
    an ``adapter_model.safetensors``, training is skipped — this makes the
    scaling ablation script restart-safe.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    output_dir = Path(output_dir)
    if resume_if_exists and (output_dir / "adapter_model.safetensors").exists():
        print(f"[train] adapter already exists at {output_dir} — skipping")
        return output_dir

    bnb = BitsAndBytesConfig(
        load_in_4bit=quant_cfg.load_in_4bit,
        bnb_4bit_quant_type=quant_cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=quant_cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg.bnb_4bit_compute_dtype),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pad_id = tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")
    if pad_id is not None and pad_id != tokenizer.unk_token_id:
        tokenizer.pad_token_id = pad_id
    elif tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
    )
    base = prepare_model_for_kbit_training(
        base, use_gradient_checkpointing=train_cfg.gradient_checkpointing
    )

    lora = LoraConfig(
        r=qlora_cfg.r,
        lora_alpha=qlora_cfg.lora_alpha,
        lora_dropout=qlora_cfg.lora_dropout,
        bias=qlora_cfg.bias,
        task_type=qlora_cfg.task_type,
        target_modules=list(qlora_cfg.target_modules),
    )
    model = get_peft_model(base, lora)

    sft_cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.num_train_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        warmup_ratio=train_cfg.warmup_ratio,
        weight_decay=train_cfg.weight_decay,
        max_grad_norm=train_cfg.max_grad_norm,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        max_seq_length=train_cfg.max_seq_length,
        logging_steps=train_cfg.logging_steps,
        save_strategy=train_cfg.save_strategy,
        fp16=train_cfg.fp16,
        bf16=train_cfg.bf16,
        seed=train_cfg.seed,
        report_to="tensorboard",
    )

    ds = Dataset.from_list(train_records)
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    return output_dir
