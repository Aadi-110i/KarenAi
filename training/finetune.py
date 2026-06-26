#!/usr/bin/env python3
"""QLoRA fine-tuning script for Karen AI.

Supports fine-tuning multiple model families (Llama 3.1 8B, Phi-3.5 Mini,
Mistral 7B, Gemma 2 9B) using 4-bit quantization with BitsAndBytes NF4
and LoRA adapters via PEFT. Uses SFTTrainer from the trl library.

Usage:
    python finetune.py --model llama-3.1-8b --dataset-dir data/processed
    python finetune.py --model mistral-7b --use-wandb --wandb-project karen-ai
    python finetune.py --model meta-llama/Llama-3.1-8B --epochs 5
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from datasets import load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from karen_config import (
    LORA_TARGET_MODULES,
    MODEL_CHOICES,
    PathConfig,
    TrainingConfig,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("karen.finetune")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_model_name(model_key: str) -> str:
    """Resolve a friendly model key to a HuggingFace model ID.

    If *model_key* is already a full HuggingFace path (contains ``/``), it is
    returned unchanged.  Otherwise it is looked up in ``MODEL_CHOICES``.

    Args:
        model_key: Short key (e.g. ``"llama-3.1-8b"``) or full HF path.

    Returns:
        The HuggingFace model identifier.

    Raises:
        ValueError: If *model_key* is not found in ``MODEL_CHOICES``.
    """
    if "/" in model_key:
        return model_key
    if model_key not in MODEL_CHOICES:
        available = ", ".join(sorted(MODEL_CHOICES.keys()))
        raise ValueError(
            f"Unknown model key '{model_key}'. "
            f"Available choices: {available}. "
            f"You may also pass a full HuggingFace model path (e.g. 'org/model')."
        )
    return MODEL_CHOICES[model_key]


def get_target_modules(model_name: str) -> list[str]:
    """Return LoRA target modules for a given model.

    Looks up the model in ``LORA_TARGET_MODULES`` by matching against
    known model-family substrings.

    Args:
        model_name: The full HuggingFace model identifier.

    Returns:
        A list of module name strings to target with LoRA.
    """
    model_lower = model_name.lower()
    for family_key, modules in LORA_TARGET_MODULES.items():
        if family_key.lower() in model_lower:
            logger.info(
                "Using LoRA target modules for family '%s': %s",
                family_key,
                modules,
            )
            return modules

    # Fallback – common attention projection names work for most architectures.
    default_modules = ["q_proj", "v_proj"]
    logger.warning(
        "No specific LoRA target modules found for '%s'. "
        "Falling back to defaults: %s",
        model_name,
        default_modules,
    )
    return default_modules


def detect_dtype() -> tuple[bool, bool]:
    """Auto-detect whether to use bf16 or fp16 based on GPU capability.

    Returns:
        A ``(use_bf16, use_fp16)`` tuple.
    """
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        logger.info("BF16 supported — training with bf16 precision.")
        return True, False
    logger.info("BF16 not supported — falling back to fp16 precision.")
    return False, True


# ---------------------------------------------------------------------------
# Core training logic
# ---------------------------------------------------------------------------


def load_quantized_model(
    model_name: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a model with 4-bit NF4 quantization via BitsAndBytes.

    The model is loaded in 4-bit precision using the NF4 data type with
    double quantization enabled to further reduce memory usage. Compute
    operations are performed in ``bfloat16`` (or ``float16`` if unsupported).

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        A ``(model, tokenizer)`` tuple.
    """
    logger.info("Loading model '%s' with 4-bit NF4 quantization …", model_name)

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    logger.info("Model and tokenizer loaded successfully.")
    return model, tokenizer


def apply_lora(
    model: AutoModelForCausalLM,
    model_name: str,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float = 0.05,
) -> PeftModel:
    """Apply LoRA adapters to the quantized model.

    Args:
        model: The base quantized model.
        model_name: HuggingFace model identifier (used to look up target modules).
        lora_rank: The LoRA rank (``r``).
        lora_alpha: The LoRA alpha scaling factor.
        lora_dropout: Dropout probability for LoRA layers.

    Returns:
        The PEFT-wrapped model with LoRA adapters attached.
    """
    target_modules = get_target_modules(model_name)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        "LoRA applied — trainable parameters: %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100.0 * trainable / total,
    )
    return model


def run_training(args: argparse.Namespace) -> None:
    """Execute the full QLoRA fine-tuning pipeline.

    Steps:
        1. Resolve model name and load dataset.
        2. Load model with 4-bit quantization.
        3. Apply LoRA adapters.
        4. Configure training arguments.
        5. Train with ``SFTTrainer``.
        6. Save the LoRA adapter and tokenizer.

    Args:
        args: Parsed CLI arguments.
    """
    # ------------------------------------------------------------------
    # 1. Resolve model & load dataset
    # ------------------------------------------------------------------
    model_name = resolve_model_name(args.model)
    logger.info("Resolved model: %s", model_name)

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}. "
            "Run the data preparation pipeline first."
        )

    logger.info("Loading dataset from '%s' …", dataset_dir)
    dataset = load_from_disk(str(dataset_dir))
    logger.info("Dataset loaded — %d examples.", len(dataset))

    # ------------------------------------------------------------------
    # 2. Load quantized model
    # ------------------------------------------------------------------
    model, tokenizer = load_quantized_model(model_name)

    # ------------------------------------------------------------------
    # 3. Apply LoRA
    # ------------------------------------------------------------------
    model = apply_lora(
        model,
        model_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )

    # Enable gradient checkpointing to reduce VRAM usage.
    model.gradient_checkpointing_enable()

    # ------------------------------------------------------------------
    # 4. Configure training arguments
    # ------------------------------------------------------------------
    use_bf16, use_fp16 = detect_dtype()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir = output_dir / "logs"
    logging_dir.mkdir(parents=True, exist_ok=True)

    # Optional W&B integration
    report_to: str | list[str] = "none"
    wandb_env: dict[str, str] = {}
    if args.use_wandb:
        report_to = "wandb"
        wandb_env = {
            "WANDB_PROJECT": args.wandb_project,
            "WANDB_LOG_MODEL": "false",
        }
        for key, value in wandb_env.items():
            os.environ[key] = value
        logger.info(
            "Weights & Biases enabled — project: '%s'", args.wandb_project
        )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 16 // args.batch_size),
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_dir=str(logging_dir),
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=report_to,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    # ------------------------------------------------------------------
    # 5. Train with SFTTrainer
    # ------------------------------------------------------------------
    logger.info("Initializing SFTTrainer …")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        max_seq_length=args.max_seq_length,
        packing=True,
    )

    resume_ckpt: Optional[str] = None
    if args.resume_from_checkpoint:
        resume_ckpt = args.resume_from_checkpoint
        logger.info("Resuming from checkpoint: %s", resume_ckpt)

    logger.info("Starting training …")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    logger.info("Training complete.")

    # ------------------------------------------------------------------
    # 6. Save adapter & tokenizer
    # ------------------------------------------------------------------
    adapter_dir = output_dir / "final_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving LoRA adapter to '%s' …", adapter_dir)
    model.save_pretrained(str(adapter_dir))

    logger.info("Saving tokenizer to '%s' …", adapter_dir)
    tokenizer.save_pretrained(str(adapter_dir))

    logger.info(
        "✅  Fine-tuning complete. Adapter saved to: %s", adapter_dir
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the QLoRA fine-tuning script.

    Returns:
        Parsed ``argparse.Namespace`` with all configuration values.
    """
    parser = argparse.ArgumentParser(
        description="Karen AI — QLoRA Fine-Tuning Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model selection
    model_help = (
        "Model key from MODEL_CHOICES (e.g. 'llama-3.1-8b') or a full "
        "HuggingFace model path (e.g. 'meta-llama/Llama-3.1-8B')."
    )
    parser.add_argument("--model", type=str, required=True, help=model_help)

    # Data & output
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=getattr(PathConfig, "PROCESSED_DATA_DIR", "data/processed"),
        help="Directory containing the prepared HuggingFace dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=getattr(PathConfig, "SFT_OUTPUT_DIR", "output/sft"),
        help="Directory for checkpoints and training outputs.",
    )

    # Training hyper-parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=getattr(TrainingConfig, "NUM_EPOCHS", 3),
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=getattr(TrainingConfig, "BATCH_SIZE", 4),
        help="Per-device training batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=getattr(TrainingConfig, "LEARNING_RATE", 2e-4),
        help="Peak learning rate.",
    )

    # LoRA hyper-parameters
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=getattr(TrainingConfig, "LORA_RANK", 64),
        help="LoRA rank (r).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=getattr(TrainingConfig, "LORA_ALPHA", 128),
        help="LoRA alpha scaling factor.",
    )

    # Sequence length
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=getattr(TrainingConfig, "MAX_SEQ_LENGTH", 2048),
        help="Maximum sequence length for training examples.",
    )

    # Checkpoint resumption
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )

    # Weights & Biases
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases experiment tracking.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="karen-ai",
        help="Weights & Biases project name.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry-point for the Karen AI QLoRA fine-tuning script."""
    args = parse_args()
    logger.info("Karen AI — QLoRA Fine-Tuning")
    logger.info("Configuration: %s", vars(args))

    try:
        run_training(args)
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)
    except torch.cuda.OutOfMemoryError:
        logger.error(
            "CUDA out of memory! Try one or more of the following:\n"
            "  • Reduce --batch-size (current: %d)\n"
            "  • Reduce --max-seq-length (current: %d)\n"
            "  • Reduce --lora-rank (current: %d)\n"
            "  • Use a smaller model\n"
            "  • Free GPU memory by closing other processes",
            args.batch_size,
            args.max_seq_length,
            args.lora_rank,
        )
        sys.exit(1)
    except Exception:
        logger.exception("Unexpected error during training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
