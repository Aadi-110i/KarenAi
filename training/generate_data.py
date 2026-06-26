#!/usr/bin/env python3
"""Synthetic data generation script for Karen AI training pipeline.

Generates multi-turn training conversations for Karen AI — an AI companion
helping kids aged 9–16 navigate puberty and adolescence. Connects to an
OpenAI-compatible API and produces JSONL output suitable for fine-tuning.

Usage:
    python generate_data.py --num-samples 200 --model llama3.1
    python generate_data.py --api-url http://localhost:11434/v1 --seed-file seeds.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import openai
from tqdm import tqdm

from karen_config import (
    AGE_GROUPS,
    GENDER_CONTEXTS,
    KAREN_SYSTEM_PROMPT,
    TOPIC_CATEGORIES,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_data")

# ---------------------------------------------------------------------------
# Meta-prompts used to steer the generation model
# ---------------------------------------------------------------------------
GENERATION_PROMPTS: dict[str, list[str]] = {
    "puberty_body_changes": [
        (
            "Generate a realistic conversation where a {age_group} ({age_range}) "
            "{gender} asks Karen about physical changes they are experiencing during "
            "puberty. Karen should be warm, age-appropriate, and factual."
        ),
        (
            "Create a multi-turn dialogue where a {age_group} {gender} is confused "
            "or worried about body changes (growth spurts, voice changes, body hair, "
            "etc.). Karen reassures them while providing accurate health information."
        ),
        (
            "Write a conversation where a {age_group} {gender} compares themselves "
            "to peers regarding development. Karen helps them understand that everyone "
            "develops at their own pace."
        ),
        (
            "Simulate a chat where a {age_group} {gender} asks Karen about puberty "
            "milestones they have read about online and wants to know what is normal. "
            "Karen provides clear, supportive answers."
        ),
    ],
    "emotions_mental_health": [
        (
            "Generate a conversation where a {age_group} ({age_range}) {gender} "
            "talks to Karen about mood swings and intense emotions. Karen validates "
            "their feelings and offers coping strategies."
        ),
        (
            "Create a dialogue where a {age_group} {gender} feels overwhelmed by "
            "stress from school or home life. Karen helps them break the problem "
            "down and suggests practical self-care."
        ),
        (
            "Write a multi-turn chat where a {age_group} {gender} opens up about "
            "feeling anxious or sad. Karen listens empathetically and gently "
            "encourages talking to a trusted adult."
        ),
        (
            "Simulate a conversation where a {age_group} {gender} is dealing with "
            "low self-esteem during adolescence. Karen helps them recognize their "
            "strengths and build confidence."
        ),
        (
            "Generate a chat where a {age_group} {gender} asks Karen why they cry "
            "easily or get angry over small things. Karen normalizes these hormonal "
            "changes and teaches emotional regulation."
        ),
    ],
    "social_relationships": [
        (
            "Generate a conversation where a {age_group} ({age_range}) {gender} "
            "asks Karen for advice on friendship drama or peer pressure. Karen "
            "coaches them on assertive communication."
        ),
        (
            "Create a dialogue where a {age_group} {gender} has questions about "
            "crushes, romantic feelings, or fitting in. Karen responds in a "
            "supportive, age-appropriate manner."
        ),
        (
            "Write a chat where a {age_group} {gender} is experiencing bullying "
            "or social exclusion. Karen provides compassionate support and "
            "actionable steps."
        ),
        (
            "Simulate a conversation where a {age_group} {gender} wants to "
            "understand changing family dynamics during adolescence. Karen "
            "helps them navigate communication with parents or siblings."
        ),
    ],
    "hygiene_self_care": [
        (
            "Generate a conversation where a {age_group} ({age_range}) {gender} "
            "asks Karen about new hygiene routines needed during puberty (deodorant, "
            "skincare, etc.). Karen gives practical, shame-free advice."
        ),
        (
            "Create a multi-turn dialogue where a {age_group} {gender} is "
            "embarrassed about body odour or acne. Karen normalizes it and "
            "suggests a simple self-care routine."
        ),
        (
            "Write a conversation where a {age_group} {gender} asks Karen about "
            "nutrition, sleep, and exercise during adolescence. Karen explains "
            "why these matter for their growing body."
        ),
    ],
    "safety_boundaries": [
        (
            "Generate a conversation where a {age_group} ({age_range}) {gender} "
            "asks Karen about personal boundaries and body autonomy. Karen teaches "
            "the concept of consent in an age-appropriate way."
        ),
        (
            "Create a dialogue where a {age_group} {gender} encounters an "
            "uncomfortable situation online. Karen helps them understand online "
            "safety rules and when to tell a trusted adult."
        ),
        (
            "Write a multi-turn chat where a {age_group} {gender} is unsure "
            "about what counts as appropriate vs. inappropriate touch. Karen "
            "explains clearly and empowers them to speak up."
        ),
        (
            "Simulate a conversation where a {age_group} {gender} asks Karen "
            "about peer pressure to share personal photos or information. Karen "
            "helps them understand risks and how to say no."
        ),
    ],
}

# ---------------------------------------------------------------------------
# Retry / back-off helpers
# ---------------------------------------------------------------------------
MAX_RETRIES: int = 5
INITIAL_BACKOFF_SECONDS: float = 1.0


def _call_with_retry(
    client: openai.OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
) -> str | None:
    """Call the chat-completions endpoint with exponential back-off.

    Returns the assistant's response text, or ``None`` on persistent failure.
    """
    backoff = INITIAL_BACKOFF_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            return content.strip() if content else None
        except openai.APIConnectionError as exc:
            logger.warning("Connection error (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
        except openai.RateLimitError as exc:
            logger.warning("Rate-limited (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
        except openai.APIStatusError as exc:
            logger.error("API status error %d (attempt %d/%d): %s", exc.status_code, attempt, MAX_RETRIES, exc)
            if exc.status_code < 500:
                # Client errors are unlikely to resolve with a retry.
                return None
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)

        if attempt < MAX_RETRIES:
            jitter = random.uniform(0, backoff * 0.5)
            sleep_time = backoff + jitter
            logger.info("Retrying in %.1f s …", sleep_time)
            time.sleep(sleep_time)
            backoff *= 2

    return None


# ---------------------------------------------------------------------------
# Conversation generation
# ---------------------------------------------------------------------------

def _build_generation_instruction(
    topic: str,
    age_group_key: str,
    gender: str,
    num_turns: int,
    few_shot_examples: list[dict[str, Any]] | None = None,
) -> str:
    """Build the meta-instruction sent to the generation model.

    Selects a random meta-prompt template for *topic*, fills it with the
    provided demographic context, and appends few-shot examples when available.
    """
    age_info: dict[str, str] = AGE_GROUPS.get(age_group_key, {})  # type: ignore[arg-type]
    age_label = age_info.get("label", age_group_key)
    age_range = age_info.get("range", "9-16")

    template = random.choice(GENERATION_PROMPTS[topic])
    prompt = template.format(
        age_group=age_label,
        age_range=age_range,
        gender=gender,
    )

    instruction_parts: list[str] = [
        "You are a data-generation assistant. Your task is to produce a "
        "realistic multi-turn conversation between a young person and Karen, "
        "an AI companion for kids aged 9-16.\n",
        f"Topic: {topic}",
        f"Age group: {age_label} ({age_range})",
        f"Gender context: {gender}",
        f"Number of exchanges (user+assistant pairs): {num_turns}\n",
        f"Scenario description:\n{prompt}\n",
        "IMPORTANT RULES:\n"
        "- Karen's responses must follow this system prompt:\n"
        f'"""\n{KAREN_SYSTEM_PROMPT}\n"""\n'
        "- Output ONLY valid JSON — a list of message objects.\n"
        '- Each message object has "role" (one of "system", "user", "assistant") '
        'and "content" (string).\n'
        '- The first message must be {"role": "system", "content": "<Karen\'s system prompt>"}.\n'
        "- Alternate user and assistant messages after the system message.\n"
        "- Do NOT include any text outside the JSON array.\n",
    ]

    if few_shot_examples:
        instruction_parts.append("Here are example conversations for reference:\n")
        for idx, example in enumerate(few_shot_examples[:3], 1):
            instruction_parts.append(f"--- Example {idx} ---")
            instruction_parts.append(json.dumps(example["messages"], indent=2))
            instruction_parts.append("")

    return "\n".join(instruction_parts)


def _parse_generated_conversation(raw: str) -> list[dict[str, str]] | None:
    """Attempt to parse the model's raw output into a message list.

    Returns ``None`` when parsing or validation fails.
    """
    # Strip markdown code fences if present.
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first and last fence lines.
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.debug("JSON parse failed for generated output.")
        return None

    # The model might return {"messages": [...]} or just [...].
    if isinstance(data, dict) and "messages" in data:
        messages = data["messages"]
    elif isinstance(data, list):
        messages = data
    else:
        logger.debug("Unexpected JSON structure: %s", type(data))
        return None

    if not isinstance(messages, list) or len(messages) < 3:
        logger.debug("Too few messages: %d", len(messages) if isinstance(messages, list) else 0)
        return None

    return messages  # type: ignore[return-value]


def _validate_conversation(messages: list[dict[str, str]]) -> bool:
    """Validate conversation structure and content quality.

    Checks:
    - Starts with a system message.
    - Alternates user / assistant after the system message.
    - All messages have non-empty ``content``.
    - At least one user and one assistant message.
    """
    if not messages:
        return False

    # Role order check.
    if messages[0].get("role") != "system":
        return False

    expected_role = "user"
    user_count = 0
    assistant_count = 0

    for msg in messages[1:]:
        role = msg.get("role")
        content = msg.get("content", "").strip()

        if not content:
            return False

        if role not in ("user", "assistant"):
            return False

        if role != expected_role:
            return False

        if role == "user":
            user_count += 1
            expected_role = "assistant"
        else:
            assistant_count += 1
            expected_role = "user"

    return user_count >= 1 and assistant_count >= 1


def _conversation_hash(messages: list[dict[str, str]]) -> str:
    """Return a deterministic hash for duplicate detection."""
    content_blob = "||".join(
        f"{m.get('role', '')}:{m.get('content', '')}" for m in messages
    )
    return hashlib.sha256(content_blob.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Seed / few-shot loading
# ---------------------------------------------------------------------------

def load_seed_conversations(path: str | Path) -> list[dict[str, Any]]:
    """Load seed conversations from a JSONL file for few-shot examples.

    Each line must be a JSON object with a ``"messages"`` key.
    """
    seeds: list[dict[str, Any]] = []
    path = Path(path)
    if not path.is_file():
        logger.warning("Seed file not found: %s", path)
        return seeds

    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "messages" in obj:
                    seeds.append(obj)
                else:
                    logger.warning("Seed line %d missing 'messages' key — skipped.", lineno)
            except json.JSONDecodeError:
                logger.warning("Seed line %d is not valid JSON — skipped.", lineno)

    logger.info("Loaded %d seed conversations from %s", len(seeds), path)
    return seeds


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_conversations(
    *,
    client: openai.OpenAI,
    model: str,
    num_samples: int,
    temperature: float,
    output_path: Path,
    seed_conversations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate *num_samples* training conversations and write to *output_path*.

    Returns a statistics dictionary.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    topics = list(GENERATION_PROMPTS.keys())
    age_group_keys = list(AGE_GROUPS.keys())
    gender_options = list(GENDER_CONTEXTS)

    seen_hashes: set[str] = set()
    stats: dict[str, int] = {t: 0 for t in topics}
    total_generated = 0
    total_failures = 0

    with output_path.open("w", encoding="utf-8") as fh:
        pbar = tqdm(total=num_samples, desc="Generating conversations", unit="conv")

        while total_generated < num_samples:
            topic = random.choice(topics)
            age_group_key = random.choice(age_group_keys)
            gender = random.choice(gender_options)
            num_turns = random.randint(2, 4)

            instruction = _build_generation_instruction(
                topic=topic,
                age_group_key=age_group_key,
                gender=gender,
                num_turns=num_turns,
                few_shot_examples=seed_conversations,
            )

            raw = _call_with_retry(
                client,
                model=model,
                messages=[{"role": "user", "content": instruction}],
                temperature=temperature,
            )

            if raw is None:
                total_failures += 1
                logger.debug("API call returned None for topic=%s", topic)
                continue

            messages = _parse_generated_conversation(raw)
            if messages is None:
                total_failures += 1
                logger.debug("Parse failure for topic=%s", topic)
                continue

            # Inject canonical system prompt if the model deviated.
            messages[0] = {"role": "system", "content": KAREN_SYSTEM_PROMPT}

            if not _validate_conversation(messages):
                total_failures += 1
                logger.debug("Validation failure for topic=%s", topic)
                continue

            conv_hash = _conversation_hash(messages)
            if conv_hash in seen_hashes:
                total_failures += 1
                logger.debug("Duplicate conversation detected — skipped.")
                continue
            seen_hashes.add(conv_hash)

            record = {"messages": messages}
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()

            stats[topic] += 1
            total_generated += 1
            pbar.update(1)

        pbar.close()

    return {
        "total_generated": total_generated,
        "total_failures": total_failures,
        "per_topic": stats,
        "output_file": str(output_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for Karen AI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:11434/v1",
        help="Base URL of the OpenAI-compatible API.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="ollama",
        help="API key for the generation endpoint.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.1",
        help="Model name to use for generation.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of conversations to generate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/generated_conversations.jsonl",
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--seed-file",
        type=str,
        default=None,
        help="Path to a JSONL file with seed conversations for few-shot examples.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the synthetic data generation script."""
    args = parse_args(argv)

    logger.info("=== Karen AI — Synthetic Data Generator ===")
    logger.info("API URL  : %s", args.api_url)
    logger.info("Model    : %s", args.model)
    logger.info("Samples  : %d", args.num_samples)
    logger.info("Output   : %s", args.output)
    logger.info("Temp     : %.2f", args.temperature)

    client = openai.OpenAI(
        base_url=args.api_url,
        api_key=args.api_key,
    )

    # Optionally load seed conversations for few-shot prompting.
    seed_conversations: list[dict[str, Any]] | None = None
    if args.seed_file:
        seed_conversations = load_seed_conversations(args.seed_file)
        if not seed_conversations:
            logger.warning("No valid seed conversations loaded — proceeding without few-shot examples.")
            seed_conversations = None

    output_path = Path(args.output)

    stats = generate_conversations(
        client=client,
        model=args.model,
        num_samples=args.num_samples,
        temperature=args.temperature,
        output_path=output_path,
        seed_conversations=seed_conversations,
    )

    # ---- Summary ----
    logger.info("=== Generation Complete ===")
    logger.info("Total generated : %d", stats["total_generated"])
    logger.info("Total failures  : %d", stats["total_failures"])
    logger.info("Output file     : %s", stats["output_file"])
    logger.info("Per-topic breakdown:")
    for topic, count in stats["per_topic"].items():
        logger.info("  %-30s %d", topic, count)


if __name__ == "__main__":
    main()
