"""
Karen AI Training Pipeline — Central Configuration
====================================================
All training hyperparameters, model choices, path management,
system prompts, and content taxonomy live here.

Every setting can be overridden via environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


# ---------------------------------------------------------------------------
# Directory anchor — everything is relative to the *training/* folder
# ---------------------------------------------------------------------------
_TRAINING_DIR = Path(__file__).resolve().parent


# ═══════════════════════════════════════════════════════════════════════════
# Model Choices
# ═══════════════════════════════════════════════════════════════════════════

MODEL_CHOICES: Dict[str, str] = {
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "phi3.5-mini": "microsoft/Phi-3.5-mini-instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma2-9b": "google/gemma-2-9b-it",
}

# The model key used by default (override with env var KAREN_MODEL_KEY)
DEFAULT_MODEL_KEY: str = os.getenv("KAREN_MODEL_KEY", "llama3.1-8b")


def get_model_id(key: str | None = None) -> str:
    """Return the HuggingFace model ID for *key* (defaults to DEFAULT_MODEL_KEY)."""
    key = key or DEFAULT_MODEL_KEY
    if key not in MODEL_CHOICES:
        raise ValueError(
            f"Unknown model key '{key}'. Choose from: {list(MODEL_CHOICES.keys())}"
        )
    return MODEL_CHOICES[key]


# ═══════════════════════════════════════════════════════════════════════════
# LoRA target modules per model family
# ═══════════════════════════════════════════════════════════════════════════

LORA_TARGET_MODULES: Dict[str, List[str]] = {
    "llama3.1-8b": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "phi3.5-mini": [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ],
    "mistral-7b": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "gemma2-9b": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# Training Configuration
# ═══════════════════════════════════════════════════════════════════════════

def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass
class TrainingConfig:
    """Hyperparameters for fine-tuning."""

    # Optimizer / scheduler
    learning_rate: float = field(
        default_factory=lambda: _env_float("KAREN_LEARNING_RATE", 2e-4)
    )
    num_epochs: int = field(
        default_factory=lambda: _env_int("KAREN_NUM_EPOCHS", 3)
    )
    per_device_train_batch_size: int = field(
        default_factory=lambda: _env_int("KAREN_BATCH_SIZE", 4)
    )
    gradient_accumulation_steps: int = field(
        default_factory=lambda: _env_int("KAREN_GRAD_ACCUM_STEPS", 4)
    )
    warmup_ratio: float = field(
        default_factory=lambda: _env_float("KAREN_WARMUP_RATIO", 0.03)
    )
    max_seq_length: int = field(
        default_factory=lambda: _env_int("KAREN_MAX_SEQ_LENGTH", 2048)
    )

    # LoRA
    lora_rank: int = field(
        default_factory=lambda: _env_int("KAREN_LORA_RANK", 64)
    )
    lora_alpha: int = field(
        default_factory=lambda: _env_int("KAREN_LORA_ALPHA", 128)
    )
    lora_dropout: float = field(
        default_factory=lambda: _env_float("KAREN_LORA_DROPOUT", 0.05)
    )

    # Model selection
    model_key: str = field(
        default_factory=lambda: os.getenv("KAREN_MODEL_KEY", DEFAULT_MODEL_KEY)
    )

    @property
    def model_id(self) -> str:
        return get_model_id(self.model_key)

    @property
    def lora_target_modules(self) -> List[str]:
        return LORA_TARGET_MODULES.get(
            self.model_key,
            LORA_TARGET_MODULES["llama3.1-8b"],  # sensible fallback
        )


# ═══════════════════════════════════════════════════════════════════════════
# Path Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PathConfig:
    """All filesystem paths — relative to the training/ directory."""

    data_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("KAREN_DATA_DIR", str(_TRAINING_DIR / "data"))
        )
    )
    output_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("KAREN_OUTPUT_DIR", str(_TRAINING_DIR / "output"))
        )
    )
    checkpoints_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("KAREN_CHECKPOINTS_DIR", str(_TRAINING_DIR / "checkpoints"))
        )
    )
    logs_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("KAREN_LOGS_DIR", str(_TRAINING_DIR / "logs"))
        )
    )
    export_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("KAREN_EXPORT_DIR", str(_TRAINING_DIR / "export"))
        )
    )

    def ensure_dirs(self) -> None:
        """Create every configured directory if it doesn't exist yet."""
        for p in (
            self.data_dir,
            self.output_dir,
            self.checkpoints_dir,
            self.logs_dir,
            self.export_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Karen's System Prompt
# ═══════════════════════════════════════════════════════════════════════════

KAREN_SYSTEM_PROMPT: str = (
    "You are Karen — a warm, supportive, and knowledgeable AI companion "
    "designed specifically for young people aged 9-16 who are navigating "
    "puberty and adolescence.\n\n"
    "## Your Role\n"
    "- You are like a caring older sister or a trusted school counselor "
    "who genuinely understands what growing up feels like.\n"
    "- You answer questions about body changes, emotions, hygiene, "
    "relationships, and self-care with honesty, warmth, and zero judgment.\n"
    "- You normalize every experience — there is no 'weird' or 'wrong' "
    "when it comes to growing up.\n\n"
    "## Your Tone & Language\n"
    "- Warm, friendly, and conversational — never clinical or preachy.\n"
    "- Use age-appropriate language: simpler for tweens (9-12), slightly "
    "more mature for teens (13-16), but always clear.\n"
    "- Sprinkle in gentle encouragement and light humor when appropriate.\n"
    "- Validate feelings first, then provide helpful information.\n"
    "- Use inclusive language that respects all genders and backgrounds.\n\n"
    "## Your Boundaries\n"
    "- NEVER provide medical diagnoses or replace professional medical advice. "
    "Always encourage talking to a doctor, nurse, or trusted adult for "
    "health concerns.\n"
    "- NEVER discuss explicit sexual content. Keep all explanations "
    "age-appropriate and educational.\n"
    "- If a young person expresses self-harm ideation, suicidal thoughts, "
    "or describes abuse, respond with empathy and IMMEDIATELY provide "
    "crisis resources (Crisis Text Line: text HOME to 741741, "
    "988 Suicide & Crisis Lifeline: call/text 988, "
    "Childhelp National Child Abuse Hotline: 1-800-422-4453) and "
    "strongly encourage them to tell a trusted adult right away.\n"
    "- If someone appears to be an adult pretending to be a child, or "
    "the conversation feels unsafe, gently disengage and suggest "
    "speaking with a trusted adult.\n"
    "- Stay within your scope: puberty, adolescence, emotional well-being, "
    "hygiene, social skills, and safety. For off-topic requests, kindly "
    "redirect.\n\n"
    "## How You Help\n"
    "- Break down complex topics into simple, digestible pieces.\n"
    "- Share that millions of other kids go through the same things.\n"
    "- Offer practical tips and actionable advice.\n"
    "- Encourage open communication with parents, guardians, or other "
    "trusted adults.\n"
    "- Celebrate the user's courage in asking questions — it takes bravery "
    "to talk about this stuff!\n"
)


# ═══════════════════════════════════════════════════════════════════════════
# Content Taxonomy
# ═══════════════════════════════════════════════════════════════════════════

AGE_GROUPS: Dict[str, Dict[str, str]] = {
    "tween": {
        "range": "9-12",
        "description": (
            "Pre-teens just beginning puberty. They need simple, reassuring "
            "explanations and lots of normalization. Conversations should be "
            "gentle and use concrete, everyday language."
        ),
    },
    "teen": {
        "range": "13-16",
        "description": (
            "Teenagers in the midst of puberty and early adolescence. They "
            "can handle more nuanced discussions, want to feel respected as "
            "young adults, and value honesty and relatability."
        ),
    },
}

TOPIC_CATEGORIES: List[str] = [
    "puberty_body_changes",
    "emotions_mental_health",
    "social_relationships",
    "hygiene_self_care",
    "safety_boundaries",
]

GENDER_CONTEXTS: List[str] = [
    "boys",
    "girls",
    "all",
]


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: default singletons
# ═══════════════════════════════════════════════════════════════════════════

training_config = TrainingConfig()
path_config = PathConfig()
