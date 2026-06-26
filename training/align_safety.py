#!/usr/bin/env python3
"""DPO safety alignment script for Karen AI.

Trains a fine-tuned Karen AI model with Direct Preference Optimization (DPO)
to align responses toward safe, empathetic, and appropriate behaviour while
penalising unsafe, dismissive, or inappropriate outputs.

Supports both pre-made DPO pair datasets (JSONL with prompt/chosen/rejected
fields) and automatic generation of DPO pairs from safety test prompts.

Usage:
    python align_safety.py --model-path output/sft/final_adapter --base-model llama-3.1-8b
    python align_safety.py --model-path output/sft/final_adapter --base-model llama-3.1-8b \\
        --generate-pairs --safety-tests data/safety_tests.jsonl
    python align_safety.py --model-path output/sft/final_adapter --base-model llama-3.1-8b \\
        --dpo-data data/dpo_pairs.jsonl --beta 0.15
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

from karen_config import MODEL_CHOICES, PathConfig, TrainingConfig

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("karen.align_safety")

# ---------------------------------------------------------------------------
# Synthetic rejected-response templates
# ---------------------------------------------------------------------------

_REJECTED_TEMPLATES: dict[str, list[str]] = {
    "medical_advice": [
        (
            "Based on your symptoms, you likely have {condition}. "
            "I recommend you take {medication} twice a day and you should "
            "be fine within a week."
        ),
        (
            "That sounds like {condition} to me. You should start taking "
            "{medication} immediately — no need to see a doctor for this."
        ),
        (
            "I've seen this before — it's clearly {condition}. Skip the "
            "doctor visit and just pick up some {medication} from the pharmacy."
        ),
    ],
    "dismissive": [
        "That doesn't sound like a big deal. You'll get over it.",
        "I don't see why you're so upset about this. Just move on.",
        "Everyone goes through this. Stop overthinking it.",
        "You're being dramatic. It's really not that serious.",
        "Honestly, I don't think this is worth talking about.",
    ],
    "inappropriate_content": [
        (
            "Here's a detailed guide on how to {unsafe_action}. "
            "First, you'll need to…"
        ),
        (
            "Sure, I can help with that! Here's exactly how to "
            "{unsafe_action} step by step."
        ),
        (
            "No problem! To {unsafe_action}, you should start by gathering "
            "the following materials…"
        ),
    ],
    "too_clinical": [
        (
            "Your described symptomatology is consistent with a generalised "
            "anxiety disorder (ICD-10 F41.1). Cognitive-behavioural therapy "
            "has an effect size of d=0.82 in meta-analytic reviews."
        ),
        (
            "The presented case aligns with major depressive disorder, "
            "single episode, moderate (DSM-5 296.22). SSRIs demonstrate "
            "NNT of 7-8 in RCTs."
        ),
        (
            "Statistically, your situation falls within the 68th percentile "
            "of reported stressors. Recommend structured problem-solving "
            "therapy per NICE guidelines."
        ),
    ],
}

_PLACEHOLDER_CONDITIONS = [
    "acute bronchitis",
    "a mild thyroid imbalance",
    "early-stage anaemia",
    "a vitamin D deficiency",
    "a sinus infection",
]

_PLACEHOLDER_MEDICATIONS = [
    "amoxicillin 500mg",
    "ibuprofen 400mg",
    "levothyroxine",
    "a high-dose vitamin supplement",
    "prednisone",
]

_PLACEHOLDER_UNSAFE_ACTIONS = [
    "bypass safety restrictions",
    "access restricted content",
    "manipulate someone's feelings",
    "get prescription drugs without a prescription",
    "create misleading medical advice",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_model_name(model_key: str) -> str:
    """Resolve a friendly model key to a HuggingFace model ID.

    If *model_key* already contains ``/`` it is treated as a full HuggingFace
    path.  Otherwise it is looked up in ``MODEL_CHOICES``.

    Args:
        model_key: Short key or full HF model path.

    Returns:
        Resolved HuggingFace model identifier.

    Raises:
        ValueError: If the key is not found in ``MODEL_CHOICES``.
    """
    if "/" in model_key:
        return model_key
    if model_key not in MODEL_CHOICES:
        available = ", ".join(sorted(MODEL_CHOICES.keys()))
        raise ValueError(
            f"Unknown model key '{model_key}'. Available: {available}. "
            "You may also pass a full HuggingFace model path."
        )
    return MODEL_CHOICES[model_key]


def _fill_template(template: str) -> str:
    """Replace placeholders in a rejected-response template.

    Args:
        template: Template string with ``{condition}``, ``{medication}``,
            or ``{unsafe_action}`` placeholders.

    Returns:
        A filled-in string with randomly-selected placeholders.
    """
    return template.format(
        condition=random.choice(_PLACEHOLDER_CONDITIONS),
        medication=random.choice(_PLACEHOLDER_MEDICATIONS),
        unsafe_action=random.choice(_PLACEHOLDER_UNSAFE_ACTIONS),
    )


def create_synthetic_rejected(category: Optional[str] = None) -> str:
    """Generate a single synthetic rejected response.

    Rejected responses simulate unsafe, inappropriate, dismissive, or overly
    clinical outputs that Karen should *not* produce.

    Args:
        category: One of ``"medical_advice"``, ``"dismissive"``,
            ``"inappropriate_content"``, ``"too_clinical"``.  If ``None``, a
            random category is chosen.

    Returns:
        A synthetic rejected response string.
    """
    if category is None:
        category = random.choice(list(_REJECTED_TEMPLATES.keys()))
    templates = _REJECTED_TEMPLATES.get(category, _REJECTED_TEMPLATES["dismissive"])
    template = random.choice(templates)
    return _fill_template(template)


# ---------------------------------------------------------------------------
# DPO pair generation
# ---------------------------------------------------------------------------


def generate_dpo_pairs_from_safety_tests(
    safety_tests_path: Path,
) -> list[dict[str, str]]:
    """Generate DPO training pairs from a safety-tests JSONL file.

    Each line in the JSONL file is expected to have at least a ``"prompt"``
    field and optionally a ``"safe_response"`` field.  If ``"safe_response"``
    is absent a default empathetic placeholder is used as the chosen response.
    Rejected responses are synthesised from the built-in templates.

    Args:
        safety_tests_path: Path to the ``safety_tests.jsonl`` file.

    Returns:
        A list of dicts with ``"prompt"``, ``"chosen"``, and ``"rejected"``
        keys.

    Raises:
        FileNotFoundError: If *safety_tests_path* does not exist.
    """
    if not safety_tests_path.exists():
        raise FileNotFoundError(
            f"Safety tests file not found: {safety_tests_path}"
        )

    pairs: list[dict[str, str]] = []
    categories = list(_REJECTED_TEMPLATES.keys())

    with open(safety_tests_path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping invalid JSON on line %d of %s: %s",
                    line_no,
                    safety_tests_path,
                    exc,
                )
                continue

            prompt = entry.get("prompt", "").strip()
            if not prompt:
                logger.warning(
                    "Skipping line %d — missing 'prompt' field.", line_no
                )
                continue

            chosen = entry.get(
                "safe_response",
                (
                    "I understand how you're feeling, and I want you to know "
                    "that your feelings are valid. While I'm here to listen "
                    "and support you, for this particular concern I'd really "
                    "encourage you to speak with a qualified professional who "
                    "can give you the personalised help you deserve."
                ),
            ).strip()

            # Generate rejected responses from multiple categories to provide
            # diversity within the training data.
            for category in categories:
                rejected = create_synthetic_rejected(category)
                pairs.append(
                    {
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                    }
                )

    logger.info(
        "Generated %d DPO pairs from %d safety-test prompts.",
        len(pairs),
        len(pairs) // max(len(categories), 1),
    )
    return pairs


def load_dpo_pairs(dpo_data_path: Path) -> list[dict[str, str]]:
    """Load pre-made DPO pairs from a JSONL file.

    Each line must contain ``"prompt"``, ``"chosen"``, and ``"rejected"``
    fields.

    Args:
        dpo_data_path: Path to the JSONL file.

    Returns:
        A list of dicts with the three required fields.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not dpo_data_path.exists():
        raise FileNotFoundError(f"DPO data file not found: {dpo_data_path}")

    pairs: list[dict[str, str]] = []
    required_keys = {"prompt", "chosen", "rejected"}

    with open(dpo_data_path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping invalid JSON on line %d: %s", line_no, exc
                )
                continue

            missing = required_keys - entry.keys()
            if missing:
                logger.warning(
                    "Skipping line %d — missing fields: %s", line_no, missing
                )
                continue

            pairs.append(
                {
                    "prompt": entry["prompt"].strip(),
                    "chosen": entry["chosen"].strip(),
                    "rejected": entry["rejected"].strip(),
                }
            )

    logger.info("Loaded %d DPO pairs from '%s'.", len(pairs), dpo_data_path)
    return pairs


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_for_dpo(
    base_model_name: str,
    adapter_path: Path,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the SFT-finetuned model for DPO training.

    Loads the base model with 4-bit quantization and merges or loads the
    LoRA adapter on top.

    Args:
        base_model_name: HuggingFace model identifier for the base model.
        adapter_path: Path to the saved LoRA adapter directory.

    Returns:
        A ``(model, tokenizer)`` tuple ready for DPO training.

    Raises:
        FileNotFoundError: If the adapter path does not exist.
    """
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"SFT adapter not found at '{adapter_path}'. "
            "Run finetune.py first to produce the adapter."
        )

    logger.info("Loading base model '%s' with 4-bit quantization …", base_model_name)

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
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info("Loading LoRA adapter from '%s' …", adapter_path)
    model = PeftModel.from_pretrained(
        model,
        str(adapter_path),
        is_trainable=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    logger.info("Model loaded with LoRA adapter — ready for DPO training.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Core DPO training logic
# ---------------------------------------------------------------------------


def run_dpo_training(args: argparse.Namespace) -> None:
    """Execute the full DPO safety-alignment pipeline.

    Steps:
        1. Build or load DPO pairs.
        2. Load the SFT-finetuned model with LoRA adapter.
        3. Configure DPO training arguments.
        4. Train with ``DPOTrainer``.
        5. Save the aligned adapter and tokenizer.

    Args:
        args: Parsed CLI arguments.
    """
    # ------------------------------------------------------------------
    # 1. Build or load DPO pairs
    # ------------------------------------------------------------------
    if args.generate_pairs:
        safety_tests_path = Path(args.safety_tests)
        logger.info(
            "Generating DPO pairs from safety tests: %s", safety_tests_path
        )
        pairs = generate_dpo_pairs_from_safety_tests(safety_tests_path)
    else:
        if args.dpo_data is None:
            raise ValueError(
                "Either --dpo-data must be provided or --generate-pairs must "
                "be set to automatically create DPO pairs from safety tests."
            )
        dpo_data_path = Path(args.dpo_data)
        pairs = load_dpo_pairs(dpo_data_path)

    if not pairs:
        raise ValueError(
            "No DPO pairs were loaded or generated. Check your data sources."
        )

    # Convert to HuggingFace Dataset
    dpo_dataset = Dataset.from_list(pairs)
    logger.info("DPO dataset ready — %d examples.", len(dpo_dataset))

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    base_model_name = resolve_model_name(args.base_model)
    adapter_path = Path(args.model_path)
    model, tokenizer = load_model_for_dpo(base_model_name, adapter_path)

    # ------------------------------------------------------------------
    # 3. Configure DPO training
    # ------------------------------------------------------------------
    use_bf16 = (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    )
    use_fp16 = not use_bf16

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir = output_dir / "logs"
    logging_dir.mkdir(parents=True, exist_ok=True)

    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, 8 // args.batch_size),
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_dir=str(logging_dir),
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        remove_unused_columns=False,
        beta=args.beta,
    )

    # ------------------------------------------------------------------
    # 4. Train with DPOTrainer
    # ------------------------------------------------------------------
    logger.info(
        "Initializing DPOTrainer (beta=%.3f, lr=%.2e, epochs=%d) …",
        args.beta,
        args.learning_rate,
        args.epochs,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dpo_dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting DPO safety alignment training …")
    trainer.train()
    logger.info("DPO training complete.")

    # ------------------------------------------------------------------
    # 5. Save aligned adapter & tokenizer
    # ------------------------------------------------------------------
    aligned_dir = output_dir / "aligned_adapter"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving aligned adapter to '%s' …", aligned_dir)
    model.save_pretrained(str(aligned_dir))

    logger.info("Saving tokenizer to '%s' …", aligned_dir)
    tokenizer.save_pretrained(str(aligned_dir))

    logger.info(
        "✅  DPO safety alignment complete. Aligned adapter saved to: %s",
        aligned_dir,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the DPO safety alignment script.

    Returns:
        Parsed ``argparse.Namespace`` with all configuration values.
    """
    parser = argparse.ArgumentParser(
        description="Karen AI — DPO Safety Alignment Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model / adapter paths
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the SFT LoRA adapter directory (output of finetune.py).",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help=(
            "Base model key from MODEL_CHOICES or a full HuggingFace path. "
            "Must match the model used for SFT fine-tuning."
        ),
    )

    # DPO data sources
    parser.add_argument(
        "--dpo-data",
        type=str,
        default=None,
        help="Path to a JSONL file with prompt/chosen/rejected fields.",
    )
    parser.add_argument(
        "--generate-pairs",
        action="store_true",
        default=False,
        help="Auto-generate DPO pairs from safety test prompts.",
    )
    parser.add_argument(
        "--safety-tests",
        type=str,
        default=getattr(
            PathConfig, "SAFETY_TESTS_PATH", "data/safety_tests.jsonl"
        ),
        help="Path to the safety_tests.jsonl file (used with --generate-pairs).",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=getattr(PathConfig, "DPO_OUTPUT_DIR", "output/dpo"),
        help="Directory for DPO checkpoints and outputs.",
    )

    # Training hyper-parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of DPO training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=getattr(TrainingConfig, "DPO_BATCH_SIZE", 2),
        help="Per-device training batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Peak learning rate for DPO training.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO beta parameter — controls deviation from reference policy.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry-point for the Karen AI DPO safety alignment script."""
    args = parse_args()
    logger.info("Karen AI — DPO Safety Alignment")
    logger.info("Configuration: %s", vars(args))

    try:
        run_dpo_training(args)
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
            "  • Use a smaller base model\n"
            "  • Free GPU memory by closing other processes",
            args.batch_size,
        )
        sys.exit(1)
    except Exception:
        logger.exception("Unexpected error during DPO alignment.")
        sys.exit(1)


if __name__ == "__main__":
    main()
