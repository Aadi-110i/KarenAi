#!/usr/bin/env python3
"""Dataset preparation script for Karen AI training pipeline.

Loads raw JSONL conversation files, cleans and validates them, removes
duplicates, applies chat-template formatting via a HuggingFace tokenizer,
and produces a train/eval split saved as a HuggingFace ``datasets`` Dataset.

Usage:
    python prepare_dataset.py --input data/generated_conversations.jsonl
    python prepare_dataset.py --input file1.jsonl file2.jsonl --model-name meta-llama/Llama-3-8B-Instruct
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
from tqdm import tqdm
from transformers import AutoTokenizer

from karen_config import KAREN_SYSTEM_PROMPT, TOPIC_CATEGORIES

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("prepare_dataset")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_ROLES = {"system", "user", "assistant"}
MIN_CONVERSATION_MESSAGES = 3  # system + at least 1 user + 1 assistant


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_jsonl_files(paths: list[str | Path]) -> list[dict[str, Any]]:
    """Load and concatenate conversations from one or more JSONL files.

    Each line must be a JSON object with a ``"messages"`` key containing a
    list of message dicts (``role`` + ``content``).

    Returns:
        A list of raw conversation records.
    """
    records: list[dict[str, Any]] = []

    for path in paths:
        p = Path(path)
        if not p.is_file():
            logger.warning("Input file not found, skipping: %s", p)
            continue

        with p.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("%s:%d — invalid JSON, skipped.", p.name, lineno)
                    continue

                if "messages" not in obj or not isinstance(obj["messages"], list):
                    logger.warning("%s:%d — missing or invalid 'messages' key, skipped.", p.name, lineno)
                    continue

                records.append(obj)

    logger.info("Loaded %d raw records from %d file(s).", len(records), len(paths))
    return records


# ---------------------------------------------------------------------------
# Cleaning & Validation
# ---------------------------------------------------------------------------

def clean_message(msg: dict[str, str]) -> dict[str, str] | None:
    """Strip whitespace from a message and validate its structure.

    Returns:
        The cleaned message dict, or ``None`` if invalid.
    """
    role = msg.get("role", "").strip().lower()
    content = msg.get("content", "")

    if isinstance(content, str):
        content = content.strip()
    else:
        return None

    if role not in VALID_ROLES:
        return None

    if not content:
        return None

    return {"role": role, "content": content}


def validate_conversation(messages: list[dict[str, str]]) -> bool:
    """Validate that a conversation follows the expected structure.

    Checks:
    - Has at least ``MIN_CONVERSATION_MESSAGES`` messages.
    - First message is a ``system`` message.
    - After the system message, roles strictly alternate: user → assistant → user → …
    - Each message has a non-empty ``content``.
    """
    if len(messages) < MIN_CONVERSATION_MESSAGES:
        return False

    if messages[0]["role"] != "system":
        return False

    expected = "user"
    for msg in messages[1:]:
        if msg["role"] != expected:
            return False
        expected = "assistant" if expected == "user" else "user"

    return True


def clean_conversations(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Clean every conversation and drop invalid ones.

    Returns:
        A list of records with cleaned messages that passed validation.
    """
    cleaned: list[dict[str, Any]] = []

    for record in tqdm(records, desc="Cleaning conversations", unit="conv"):
        raw_messages = record.get("messages", [])
        messages: list[dict[str, str]] = []

        skip = False
        for msg in raw_messages:
            cleaned_msg = clean_message(msg)
            if cleaned_msg is None:
                skip = True
                break
            messages.append(cleaned_msg)

        if skip:
            continue

        if not validate_conversation(messages):
            continue

        cleaned.append({"messages": messages})

    logger.info(
        "Cleaning: kept %d / %d conversations (%.1f%%).",
        len(cleaned),
        len(records),
        100.0 * len(cleaned) / max(len(records), 1),
    )
    return cleaned


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _content_hash(messages: list[dict[str, str]]) -> str:
    """Compute a SHA-256 hash over the concatenated role:content pairs."""
    blob = "||".join(f"{m['role']}:{m['content']}" for m in messages)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def deduplicate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate conversations based on content hash.

    Returns:
        De-duplicated list of records.
    """
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []

    for record in records:
        h = _content_hash(record["messages"])
        if h in seen:
            continue
        seen.add(h)
        unique.append(record)

    removed = len(records) - len(unique)
    if removed:
        logger.info("Deduplication: removed %d duplicate(s). %d unique remain.", removed, len(unique))
    else:
        logger.info("Deduplication: no duplicates found. %d records remain.", len(unique))

    return unique


# ---------------------------------------------------------------------------
# Chat-template formatting
# ---------------------------------------------------------------------------

def apply_chat_template(
    records: list[dict[str, Any]],
    tokenizer: AutoTokenizer,  # type: ignore[type-arg]
    max_length: int,
) -> list[dict[str, Any]]:
    """Format each conversation using the tokenizer's chat template.

    Adds a ``"text"`` field with the formatted string and ``"input_ids"``
    with the tokenized IDs. Conversations exceeding *max_length* tokens are
    dropped.

    Returns:
        Formatted records (only those within *max_length*).
    """
    formatted: list[dict[str, Any]] = []
    skipped_length = 0

    for record in tqdm(records, desc="Applying chat template", unit="conv"):
        messages = record["messages"]

        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chat template failed: %s — skipping conversation.", exc)
            continue

        input_ids = tokenizer.encode(text, add_special_tokens=False)

        if len(input_ids) > max_length:
            skipped_length += 1
            continue

        formatted.append({
            "messages": messages,
            "text": text,
            "input_ids": input_ids,
            "num_tokens": len(input_ids),
        })

    if skipped_length:
        logger.info(
            "Chat template: skipped %d conversation(s) exceeding max_length=%d.",
            skipped_length,
            max_length,
        )

    logger.info("Chat template: %d conversations formatted successfully.", len(formatted))
    return formatted


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _detect_topic(messages: list[dict[str, str]]) -> str:
    """Heuristically detect the topic of a conversation.

    Scans user/assistant content for keywords associated with each
    ``TOPIC_CATEGORIES`` entry. Falls back to ``"unknown"`` if no match
    is confident enough.
    """
    # Build simple keyword mapping from TOPIC_CATEGORIES.
    topic_keywords: dict[str, list[str]] = {
        "puberty_body_changes": [
            "puberty", "body change", "growth spurt", "voice", "body hair",
            "breast", "period", "menstruation", "development",
        ],
        "emotions_mental_health": [
            "mood", "emotion", "stress", "anxiety", "sad", "angry",
            "self-esteem", "mental health", "overwhelm", "cry",
        ],
        "social_relationships": [
            "friend", "bully", "peer pressure", "crush", "relationship",
            "social", "fitting in", "family", "sibling",
        ],
        "hygiene_self_care": [
            "hygiene", "deodorant", "acne", "skincare", "shower",
            "body odour", "sleep", "nutrition", "self-care",
        ],
        "safety_boundaries": [
            "safety", "boundary", "consent", "online safety", "touch",
            "inappropriate", "personal space", "photo",
        ],
    }

    combined_text = " ".join(
        m["content"].lower() for m in messages if m["role"] in ("user", "assistant")
    )

    scores: dict[str, int] = {}
    for topic, keywords in topic_keywords.items():
        scores[topic] = sum(1 for kw in keywords if kw in combined_text)

    if not scores or max(scores.values()) == 0:
        return "unknown"

    return max(scores, key=scores.get)  # type: ignore[arg-type]


def compute_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute and log dataset statistics.

    Returns:
        A statistics dictionary.
    """
    total = len(records)
    if total == 0:
        logger.warning("No records to compute statistics on.")
        return {"total": 0}

    lengths = [len(r["messages"]) for r in records]
    avg_length = sum(lengths) / total

    token_counts = [r.get("num_tokens", 0) for r in records]
    avg_tokens = sum(token_counts) / total if any(token_counts) else 0

    topic_counter: Counter[str] = Counter()
    for r in records:
        topic = _detect_topic(r["messages"])
        topic_counter[topic] += 1

    stats: dict[str, Any] = {
        "total_examples": total,
        "avg_conversation_length_messages": round(avg_length, 2),
        "avg_conversation_length_tokens": round(avg_tokens, 2),
        "topic_distribution": dict(topic_counter.most_common()),
    }

    logger.info("=== Dataset Statistics ===")
    logger.info("Total examples            : %d", total)
    logger.info("Avg conversation messages  : %.2f", avg_length)
    logger.info("Avg conversation tokens    : %.2f", avg_tokens)
    logger.info("Topic distribution:")
    for topic, count in topic_counter.most_common():
        logger.info("  %-30s %d (%.1f%%)", topic, count, 100.0 * count / total)

    return stats


# ---------------------------------------------------------------------------
# Dataset building & splitting
# ---------------------------------------------------------------------------

def build_dataset(
    records: list[dict[str, Any]],
    eval_split: float,
    seed: int,
) -> DatasetDict:
    """Build a HuggingFace ``DatasetDict`` with train/eval splits.

    Args:
        records: Processed conversation records (must contain ``"text"``).
        eval_split: Fraction of data to reserve for evaluation.
        seed: Random seed for reproducibility.

    Returns:
        A ``DatasetDict`` with ``"train"`` and ``"eval"`` splits.
    """
    # Prepare flat columns for the Dataset.
    texts: list[str] = []
    all_input_ids: list[list[int]] = []
    all_messages: list[str] = []  # store JSON-serialised messages
    num_tokens: list[int] = []

    for r in records:
        texts.append(r["text"])
        all_input_ids.append(r["input_ids"])
        all_messages.append(json.dumps(r["messages"], ensure_ascii=False))
        num_tokens.append(r["num_tokens"])

    dataset = Dataset.from_dict({
        "text": texts,
        "input_ids": all_input_ids,
        "messages_json": all_messages,
        "num_tokens": num_tokens,
    })

    split = dataset.train_test_split(test_size=eval_split, seed=seed)
    ds_dict = DatasetDict({
        "train": split["train"],
        "eval": split["test"],
    })

    logger.info(
        "Dataset split: train=%d, eval=%d (%.0f%% eval).",
        len(ds_dict["train"]),
        len(ds_dict["eval"]),
        eval_split * 100,
    )
    return ds_dict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare training dataset for Karen AI fine-tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more JSONL files containing raw conversations.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save the processed HuggingFace dataset.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "HuggingFace model name or local path used to load the tokenizer "
            "for chat-template formatting (e.g., meta-llama/Llama-3-8B-Instruct)."
        ),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum token length per conversation. Longer ones are dropped.",
    )
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.1,
        help="Fraction of data to use for evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/eval split reproducibility.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the dataset preparation script."""
    args = parse_args(argv)

    logger.info("=== Karen AI — Dataset Preparation ===")
    logger.info("Input files : %s", ", ".join(args.input))
    logger.info("Output dir  : %s", args.output_dir)
    logger.info("Model name  : %s", args.model_name or "(none — chat template skipped)")
    logger.info("Max length  : %d tokens", args.max_length)
    logger.info("Eval split  : %.0f%%", args.eval_split * 100)
    logger.info("Seed        : %d", args.seed)

    # ---- Load ----
    records = load_jsonl_files(args.input)
    if not records:
        logger.error("No records loaded. Exiting.")
        sys.exit(1)

    # ---- Clean & validate ----
    records = clean_conversations(records)
    if not records:
        logger.error("No valid conversations after cleaning. Exiting.")
        sys.exit(1)

    # ---- Deduplicate ----
    records = deduplicate(records)

    # ---- Chat template formatting ----
    if args.model_name:
        logger.info("Loading tokenizer: %s", args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        records = apply_chat_template(records, tokenizer, args.max_length)
        if not records:
            logger.error("No conversations remain after chat-template formatting. Exiting.")
            sys.exit(1)
    else:
        # Without a tokenizer we still populate required fields.
        logger.info("No model specified — skipping chat-template formatting.")
        for r in records:
            r["text"] = json.dumps(r["messages"], ensure_ascii=False)
            r["input_ids"] = []
            r["num_tokens"] = 0

    # ---- Statistics ----
    compute_statistics(records)

    # ---- Build & save dataset ----
    ds = build_dataset(records, eval_split=args.eval_split, seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_dir))

    logger.info("Dataset saved to: %s", output_dir.resolve())
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
