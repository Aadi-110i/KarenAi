# Karen AI — Training Pipeline 🧠

> **A comprehensive guide to training Karen AI** — a warm, emotionally intelligent AI companion designed to help children aged 9–16 navigate puberty, adolescence, and growing up.

This pipeline takes you from raw seed data all the way to a deployed, safety-aligned model running locally via Ollama. Every step is reproducible, configurable, and designed with child safety as the top priority.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Pipeline Architecture](#pipeline-architecture)
5. [Step 1: Review Seed Data](#step-1-review-seed-data)
6. [Step 2: Generate Synthetic Data](#step-2-generate-synthetic-data)
7. [Step 3: Prepare Dataset](#step-3-prepare-dataset)
8. [Step 4: Fine-tune with QLoRA](#step-4-fine-tune-with-qlora)
9. [Step 5: Safety Alignment (DPO)](#step-5-safety-alignment-dpo)
10. [Step 6: Evaluate](#step-6-evaluate)
11. [Step 7: Export & Deploy](#step-7-export--deploy)
12. [Quick Start](#quick-start)
13. [Model Comparison](#model-comparison)
14. [Troubleshooting](#troubleshooting)
15. [Safety Philosophy](#safety-philosophy)
16. [Contributing](#contributing)

---

## Overview

**Karen AI** is a warm, supportive AI companion purpose-built for children aged 9–16 who are navigating the complex world of puberty, adolescence, and emotional development. She speaks with empathy, age-appropriate language, and unwavering safety boundaries.

**This training pipeline provides:**

| Stage | Purpose |
|---|---|
| **Data Generation** | Create thousands of diverse, high-quality training conversations from curated seed examples |
| **Fine-tuning** | Adapt a base language model to Karen's personality, tone, and knowledge using QLoRA |
| **Safety Alignment** | Apply DPO (Direct Preference Optimization) to reinforce safe responses and reject harmful ones |
| **Evaluation** | Automatically score the model on helpfulness, safety, tone, and age-appropriateness |
| **Deployment** | Export to GGUF format and deploy locally via Ollama for fast, private inference |

---

## Prerequisites

### Hardware

| Component | Minimum | Recommended |
|---|---|---|
| **GPU** | 16 GB VRAM (e.g., RTX 4060 Ti 16GB) | 24 GB VRAM (e.g., RTX 4090, A5000) |
| **RAM** | 16 GB | 32 GB |
| **Disk** | 50 GB free | 100 GB free |
| **Alternative** | Cloud GPU (RunPod, Vast.ai, Google Colab Pro+) | — |

### Software

| Dependency | Version | Purpose |
|---|---|---|
| **Python** | 3.10+ | Runtime |
| **CUDA** | 11.8+ | GPU acceleration |
| **Git** | 2.30+ | Version control |
| **Ollama** | Latest | Local model deployment |

> 💡 **Tip:** If you don't have a local GPU, cloud providers like [RunPod](https://runpod.io) and [Vast.ai](https://vast.ai) offer affordable A100/H100 rentals.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/KarenAi.git
cd KarenAi

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 4. Install dependencies
pip install -r training/requirements.txt
```

> ⚠️ **Note:** If you encounter issues installing `bitsandbytes` on Windows, see the [Troubleshooting](#troubleshooting) section.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Karen AI Training Pipeline                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────┐     ┌──────────────────┐     ┌───────────────┐  │
│   │  Seed Data   │────▶│ Generate Synthetic│────▶│   Prepare     │  │
│   │  (JSONL)     │     │     Data          │     │   Dataset     │  │
│   └──────────────┘     └──────────────────┘     └──────┬────────┘  │
│                                                         │           │
│                                                         ▼           │
│   ┌──────────────┐     ┌──────────────────┐     ┌───────────────┐  │
│   │  Deploy on   │◀────│  Export to GGUF   │◀────│ QLoRA         │  │
│   │  Ollama      │     │                  │     │ Fine-tune     │  │
│   └──────────────┘     └──────────────────┘     └──────┬────────┘  │
│                                                         │           │
│                                                         ▼           │
│                        ┌──────────────────┐     ┌───────────────┐  │
│                        │    Evaluate      │◀────│ DPO Safety    │  │
│                        │                  │     │ Alignment     │  │
│                        └──────────────────┘     └───────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Data flow summary:**

```
seed_conversations.jsonl
    → generate_data.py       → synthetic training data
    → prepare_dataset.py     → formatted dataset splits
    → finetune.py            → QLoRA adapter weights
    → align_safety.py        → DPO-aligned model
    → evaluate.py            → quality & safety scores
    → export_model.py        → GGUF file for Ollama
```

---

## Step 1: Review Seed Data

Seed data lives in `training/data/seed_conversations.jsonl`. Each line is a JSON object representing a single multi-turn conversation:

```json
{
  "conversations": [
    {
      "role": "system",
      "content": "You are Karen, a warm and supportive AI companion for children aged 9-16..."
    },
    {
      "role": "user",
      "content": "I'm getting hair in weird places and I'm scared"
    },
    {
      "role": "assistant",
      "content": "Oh sweetie, I totally understand why that might feel surprising or even a little scary! But I promise you — this is completely normal and it happens to everyone..."
    }
  ],
  "metadata": {
    "topic": "puberty_physical_changes",
    "age_group": "9-12",
    "sensitivity": "medium"
  }
}
```

### Adding New Seed Conversations

1. Open `training/data/seed_conversations.jsonl`
2. Add new JSON objects, one per line
3. Ensure each conversation includes:
   - A `system` message defining Karen's persona
   - At least one `user` → `assistant` exchange
   - Appropriate `metadata` tags for categorization

**Topic categories to cover:**

| Category | Examples |
|---|---|
| `puberty_physical_changes` | Body hair, growth spurts, skin changes |
| `puberty_emotional_changes` | Mood swings, new feelings, identity |
| `relationships` | Friendships, crushes, peer pressure |
| `mental_health` | Anxiety, self-esteem, stress management |
| `safety_boundaries` | Recognizing unsafe situations, trusted adults |
| `hygiene_health` | Personal care, nutrition, sleep |

> 🎯 **Goal:** Aim for at least 50–100 diverse seed conversations across all topic categories before generating synthetic data.

---

## Step 2: Generate Synthetic Data

Use `generate_data.py` to expand seed conversations into a large synthetic dataset using a teacher model (via OpenAI API or a local model).

### Basic Usage

```bash
python training/generate_data.py \
  --seed-file training/data/seed_conversations.jsonl \
  --output-file training/data/synthetic_conversations.jsonl \
  --num-samples 5000
```

### Full Options

```bash
python training/generate_data.py \
  --seed-file training/data/seed_conversations.jsonl \
  --output-file training/data/synthetic_conversations.jsonl \
  --num-samples 5000 \
  --model gpt-4o \
  --temperature 0.8 \
  --max-tokens 2048 \
  --batch-size 10 \
  --api-key $OPENAI_API_KEY \
  --topics puberty_physical_changes,puberty_emotional_changes,relationships,mental_health \
  --age-groups 9-12,13-16 \
  --max-turns 6 \
  --validate \
  --safety-filter \
  --resume
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--seed-file` | Required | Path to seed conversations JSONL |
| `--output-file` | Required | Where to save generated conversations |
| `--num-samples` | `1000` | Total number of conversations to generate |
| `--model` | `gpt-4o` | Teacher model for generation |
| `--temperature` | `0.8` | Creativity of generated conversations (0.0–1.0) |
| `--batch-size` | `10` | Number of concurrent API requests |
| `--topics` | All | Comma-separated topic filter |
| `--age-groups` | All | Target age groups to generate for |
| `--max-turns` | `6` | Maximum conversation turns per sample |
| `--validate` | `false` | Run safety validation on each sample |
| `--safety-filter` | `false` | Filter out conversations that fail safety checks |
| `--resume` | `false` | Resume from last checkpoint if interrupted |

> ⏱️ **Estimated time:** ~2–4 hours for 5,000 samples with `gpt-4o` (depends on API rate limits).

---

## Step 3: Prepare Dataset

Transform raw conversations into a format suitable for fine-tuning and DPO alignment.

### Basic Usage

```bash
python training/prepare_dataset.py \
  --input-file training/data/synthetic_conversations.jsonl \
  --output-dir training/data/prepared/
```

### Full Options

```bash
python training/prepare_dataset.py \
  --input-file training/data/synthetic_conversations.jsonl \
  --output-dir training/data/prepared/ \
  --train-split 0.9 \
  --val-split 0.05 \
  --test-split 0.05 \
  --max-length 2048 \
  --format chatml \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --generate-dpo-pairs \
  --dpo-output-dir training/data/dpo/ \
  --shuffle \
  --seed 42
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--input-file` | Required | Path to synthetic conversations JSONL |
| `--output-dir` | Required | Directory for prepared dataset splits |
| `--train-split` | `0.9` | Fraction of data for training |
| `--val-split` | `0.05` | Fraction of data for validation |
| `--test-split` | `0.05` | Fraction of data for testing |
| `--max-length` | `2048` | Maximum token length per sample |
| `--format` | `chatml` | Chat template format (`chatml`, `llama`, `mistral`) |
| `--tokenizer` | Required | HuggingFace tokenizer to use for length calculations |
| `--generate-dpo-pairs` | `false` | Also generate chosen/rejected pairs for DPO |
| `--dpo-output-dir` | `None` | Output directory for DPO preference pairs |
| `--shuffle` | `false` | Shuffle data before splitting |
| `--seed` | `42` | Random seed for reproducibility |

**Output structure:**

```
training/data/prepared/
├── train.jsonl          # Training split
├── val.jsonl            # Validation split
├── test.jsonl           # Test split
└── metadata.json        # Dataset statistics

training/data/dpo/       # If --generate-dpo-pairs is set
├── train_pairs.jsonl    # DPO training pairs
└── val_pairs.jsonl      # DPO validation pairs
```

---

## Step 4: Fine-tune with QLoRA

Fine-tune a base language model on Karen's conversational data using QLoRA (Quantized Low-Rank Adaptation) for memory-efficient training.

### Basic Usage

```bash
python training/finetune.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset training/data/prepared/ \
  --output-dir training/output/karen-v1/
```

### Full Options

```bash
python training/finetune.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset training/data/prepared/ \
  --output-dir training/output/karen-v1/ \
  --epochs 3 \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --learning-rate 2e-4 \
  --lr-scheduler cosine \
  --warmup-ratio 0.05 \
  --max-length 2048 \
  --lora-r 64 \
  --lora-alpha 128 \
  --lora-dropout 0.05 \
  --lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --quantization 4bit \
  --bf16 \
  --gradient-checkpointing \
  --logging-steps 10 \
  --save-steps 200 \
  --eval-steps 200 \
  --wandb-project karen-ai-training \
  --wandb-run-name karen-v1-llama3.1-8b \
  --seed 42
```

### Hyperparameter Guidance

| Parameter | Conservative | Balanced | Aggressive |
|---|---|---|---|
| **Learning Rate** | `1e-4` | `2e-4` | `5e-4` |
| **LoRA Rank (`r`)** | `16` | `64` | `128` |
| **LoRA Alpha** | `32` | `128` | `256` |
| **Epochs** | `1` | `3` | `5` |
| **Batch Size (effective)** | `8` | `16` | `32` |

> 💡 **Recommendation:** Start with the **Balanced** settings. If the model overfits (val loss increases), move to Conservative. If underfitting, try Aggressive.

### GPU Memory Requirements

| Model | Parameters | VRAM (QLoRA 4-bit) | VRAM (Full Fine-tune) | Training Time (5K samples) |
|---|---|---|---|---|
| Llama 3.1 8B | 8B | ~10 GB | ~64 GB | ~2–3 hours |
| Mistral 7B v0.3 | 7B | ~9 GB | ~56 GB | ~2–3 hours |
| Gemma 2 9B | 9B | ~12 GB | ~72 GB | ~3–4 hours |
| Phi-3.5 Mini | 3.8B | ~6 GB | ~30 GB | ~1–2 hours |
| Llama 3.1 70B | 70B | ~42 GB | ~560 GB | ~12–18 hours |
| Qwen 2.5 7B | 7B | ~9 GB | ~56 GB | ~2–3 hours |

> ⚠️ **Note:** VRAM estimates assume batch size 4, max sequence length 2048, and gradient checkpointing enabled. Actual usage may vary.

---

## Step 5: Safety Alignment (DPO)

Apply **Direct Preference Optimization (DPO)** to teach Karen to prefer safe, appropriate responses over harmful or inappropriate ones. This is a critical step for a children's AI.

### Why DPO Matters

DPO alignment teaches the model to:

- ✅ Redirect inappropriate questions to trusted adults
- ✅ Maintain age-appropriate language and boundaries
- ✅ Refuse to engage with harmful, predatory, or exploitative content
- ✅ Encourage professional help for serious mental health concerns
- ✅ Never role-play as a real person or authority figure

### Basic Usage

```bash
python training/align_safety.py \
  --model training/output/karen-v1/ \
  --dpo-dataset training/data/dpo/ \
  --output-dir training/output/karen-v1-aligned/
```

### Full Options

```bash
python training/align_safety.py \
  --model training/output/karen-v1/ \
  --dpo-dataset training/data/dpo/ \
  --output-dir training/output/karen-v1-aligned/ \
  --beta 0.1 \
  --epochs 1 \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --learning-rate 5e-5 \
  --max-length 2048 \
  --max-prompt-length 1024 \
  --bf16 \
  --gradient-checkpointing \
  --logging-steps 5 \
  --save-steps 100 \
  --eval-steps 100 \
  --wandb-project karen-ai-safety \
  --wandb-run-name karen-v1-dpo-alignment \
  --seed 42
```

### Key DPO Arguments

| Argument | Default | Description |
|---|---|---|
| `--beta` | `0.1` | DPO temperature — lower values create stronger preference separation |
| `--epochs` | `1` | Number of DPO training epochs (1–2 is typically sufficient) |
| `--learning-rate` | `5e-5` | Lower than SFT to avoid catastrophic forgetting |
| `--max-prompt-length` | `1024` | Maximum token length for the prompt portion |

> 🛡️ **Important:** Always review DPO training pairs manually before training. A small number of misaligned pairs can degrade safety significantly.

---

## Step 6: Evaluate

Run comprehensive evaluation across multiple dimensions: helpfulness, safety, tone, and age-appropriateness.

### Basic Usage

```bash
python training/evaluate.py \
  --model training/output/karen-v1-aligned/ \
  --test-set training/data/prepared/test.jsonl
```

### Full Options

```bash
python training/evaluate.py \
  --model training/output/karen-v1-aligned/ \
  --test-set training/data/prepared/test.jsonl \
  --output-file training/output/eval_results.json \
  --judge-model gpt-4o \
  --metrics helpfulness,safety,tone,age_appropriateness,boundary_respect \
  --num-samples 200 \
  --temperature 0.3 \
  --max-tokens 1024 \
  --batch-size 5 \
  --safety-test-suite training/data/safety_tests.jsonl \
  --verbose \
  --generate-report
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--judge-model` | `gpt-4o` | Model used to score responses |
| `--metrics` | All | Comma-separated list of evaluation dimensions |
| `--num-samples` | `100` | Number of test samples to evaluate |
| `--safety-test-suite` | `None` | Additional adversarial safety test cases |
| `--generate-report` | `false` | Generate a detailed HTML report |

### Interpreting Results

The evaluation produces scores on a 1–5 scale for each metric:

| Score | Meaning |
|---|---|
| **5** | Excellent — model response is ideal |
| **4** | Good — minor issues, acceptable for production |
| **3** | Acceptable — noticeable issues, may need more training |
| **2** | Poor — significant problems, do not deploy |
| **1** | Critical — dangerous or inappropriate response |

**Target scores for deployment:**

| Metric | Minimum Score | Target Score |
|---|---|---|
| **Helpfulness** | 3.5 | 4.0+ |
| **Safety** | 4.5 | 4.8+ |
| **Tone** | 4.0 | 4.5+ |
| **Age Appropriateness** | 4.5 | 4.8+ |
| **Boundary Respect** | 4.5 | 4.8+ |

> 🚨 **Do not deploy** if any safety-related metric falls below 4.5. Re-run DPO alignment with improved training pairs.

---

## Step 7: Export & Deploy

Convert the fine-tuned model to GGUF format and deploy it locally with Ollama.

### Export to GGUF

```bash
python training/export_model.py \
  --model training/output/karen-v1-aligned/ \
  --output-file training/output/karen-v1.gguf \
  --quantization Q4_K_M
```

### Full Export Options

```bash
python training/export_model.py \
  --model training/output/karen-v1-aligned/ \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --output-file training/output/karen-v1.gguf \
  --quantization Q4_K_M \
  --merge-adapter \
  --vocab-type spm \
  --context-length 4096 \
  --pad-vocab
```

### Quantization Options

| Quantization | Size (7B model) | Quality | Speed | Best For |
|---|---|---|---|---|
| `Q8_0` | ~7.5 GB | Highest | Slower | Maximum quality |
| `Q5_K_M` | ~5.0 GB | Very Good | Moderate | Good balance |
| `Q4_K_M` | ~4.0 GB | Good | Fast | **Recommended** |
| `Q3_K_M` | ~3.0 GB | Acceptable | Fastest | Memory-constrained |
| `Q2_K` | ~2.5 GB | Low | Fastest | Not recommended |

### Deploy on Ollama

```bash
# 1. Install Ollama (if not installed)
# Visit: https://ollama.com/download

# 2. Create a Modelfile
cat > training/output/Modelfile << 'EOF'
FROM ./karen-v1.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096

SYSTEM """You are Karen, a warm and supportive AI companion for children aged 9-16. You help them navigate puberty, adolescence, and growing up with empathy, age-appropriate language, and clear safety boundaries. You always encourage them to talk to trusted adults about serious concerns."""
EOF

# 3. Create the Ollama model
cd training/output/
ollama create karen-ai -f Modelfile

# 4. Test the model
ollama run karen-ai "Hi Karen, I'm 12 and I feel like nobody understands me"

# 5. Serve via API (for integration with the Karen AI app)
ollama serve
```

---

## Quick Start

Run the entire pipeline end-to-end with a single script:

```bash
#!/bin/bash
set -e

echo "🧠 Karen AI Training Pipeline — Quick Start"
echo "============================================="

# Configuration
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_NAME="karen-v1"
NUM_SAMPLES=5000

# Step 1: Generate synthetic data
echo "📝 Step 1/7: Generating synthetic data..."
python training/generate_data.py \
  --seed-file training/data/seed_conversations.jsonl \
  --output-file training/data/synthetic_conversations.jsonl \
  --num-samples $NUM_SAMPLES \
  --model gpt-4o \
  --validate \
  --safety-filter

# Step 2: Prepare dataset
echo "📦 Step 2/7: Preparing dataset..."
python training/prepare_dataset.py \
  --input-file training/data/synthetic_conversations.jsonl \
  --output-dir training/data/prepared/ \
  --tokenizer $BASE_MODEL \
  --generate-dpo-pairs \
  --dpo-output-dir training/data/dpo/ \
  --shuffle

# Step 3: Fine-tune with QLoRA
echo "🔧 Step 3/7: Fine-tuning with QLoRA..."
python training/finetune.py \
  --model $BASE_MODEL \
  --dataset training/data/prepared/ \
  --output-dir training/output/$OUTPUT_NAME/ \
  --epochs 3 \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --learning-rate 2e-4 \
  --lora-r 64 \
  --lora-alpha 128 \
  --bf16 \
  --gradient-checkpointing

# Step 4: Safety alignment with DPO
echo "🛡️ Step 4/7: Running DPO safety alignment..."
python training/align_safety.py \
  --model training/output/$OUTPUT_NAME/ \
  --dpo-dataset training/data/dpo/ \
  --output-dir training/output/${OUTPUT_NAME}-aligned/ \
  --beta 0.1 \
  --epochs 1 \
  --learning-rate 5e-5 \
  --bf16 \
  --gradient-checkpointing

# Step 5: Evaluate
echo "📊 Step 5/7: Evaluating model..."
python training/evaluate.py \
  --model training/output/${OUTPUT_NAME}-aligned/ \
  --test-set training/data/prepared/test.jsonl \
  --output-file training/output/eval_results.json \
  --generate-report

# Step 6: Export to GGUF
echo "📦 Step 6/7: Exporting to GGUF..."
python training/export_model.py \
  --model training/output/${OUTPUT_NAME}-aligned/ \
  --base-model $BASE_MODEL \
  --output-file training/output/${OUTPUT_NAME}.gguf \
  --quantization Q4_K_M \
  --merge-adapter

# Step 7: Deploy on Ollama
echo "🚀 Step 7/7: Deploying on Ollama..."
cd training/output/
ollama create karen-ai -f Modelfile

echo ""
echo "✅ Pipeline complete! Test your model:"
echo "   ollama run karen-ai \"Hi Karen!\""
```

Save this as `training/run_pipeline.sh` and execute:

```bash
chmod +x training/run_pipeline.sh
./training/run_pipeline.sh
```

---

## Model Comparison

| Model | Parameters | VRAM (QLoRA) | Speed | Quality | Best For |
|---|---|---|---|---|---|
| **Phi-3.5 Mini** | 3.8B | ~6 GB | ⚡ Very Fast | ★★★☆☆ | Testing, low-resource devices |
| **Mistral 7B v0.3** | 7B | ~9 GB | ⚡ Fast | ★★★★☆ | Good quality on consumer GPUs |
| **Qwen 2.5 7B** | 7B | ~9 GB | ⚡ Fast | ★★★★☆ | Strong multilingual support |
| **Llama 3.1 8B** | 8B | ~10 GB | 🔄 Moderate | ★★★★☆ | **Recommended** — best balance |
| **Gemma 2 9B** | 9B | ~12 GB | 🔄 Moderate | ★★★★☆ | Strong conversational ability |
| **Llama 3.1 70B** | 70B | ~42 GB | 🐢 Slow | ★★★★★ | Maximum quality, cloud GPUs |

> 💡 **Recommendation:** Start with **Llama 3.1 8B Instruct** for the best balance of quality, speed, and memory usage. Scale up to 70B only if you have cloud GPU access and need the highest quality.

---

## Troubleshooting

### CUDA Out of Memory (OOM)

**Symptoms:** `torch.cuda.OutOfMemoryError` or `CUDA out of memory`

**Solutions:**

```bash
# Reduce batch size
--batch-size 1 --gradient-accumulation-steps 16

# Reduce sequence length
--max-length 1024

# Use a smaller LoRA rank
--lora-r 16 --lora-alpha 32

# Enable gradient checkpointing (if not already)
--gradient-checkpointing

# Use a smaller model
--model microsoft/Phi-3.5-mini-instruct
```

### Tokenizer Errors

**Symptoms:** `TokenizerError`, `KeyError: 'chat_template'`, or garbled output

**Solutions:**

```bash
# Ensure you're using the correct tokenizer for your model
--tokenizer meta-llama/Llama-3.1-8B-Instruct

# Update transformers to latest version
pip install --upgrade transformers tokenizers

# For Llama models, ensure you have accepted the license on HuggingFace
# Visit: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
```

### OpenAI API Connection Issues

**Symptoms:** `APIConnectionError`, `RateLimitError`, or timeout errors during data generation

**Solutions:**

```bash
# Verify your API key
echo $OPENAI_API_KEY

# Reduce batch size to stay within rate limits
--batch-size 3

# Use the --resume flag to continue after interruption
--resume

# Set explicit timeout
--timeout 120
```

### GGUF Conversion Failures

**Symptoms:** `ValueError` during export, corrupted GGUF file, or Ollama refusing to load

**Solutions:**

```bash
# Ensure adapter is merged before conversion
--merge-adapter

# Try a different quantization level
--quantization Q5_K_M

# Update llama.cpp conversion tools
pip install --upgrade llama-cpp-python

# Verify the exported file
python -c "import struct; f=open('karen-v1.gguf','rb'); print(f.read(4))"
# Should print: b'GGUF'
```

### bitsandbytes on Windows

**Symptoms:** `ImportError` or DLL errors when importing bitsandbytes

**Solutions:**

```bash
# Install the Windows-compatible version
pip install bitsandbytes-windows

# Or use the pre-built wheel
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.43.0-py3-none-win_amd64.whl
```

### Weights & Biases (wandb) Issues

**Symptoms:** `wandb.errors.CommError` or login prompts during training

**Solutions:**

```bash
# Login to wandb
wandb login

# Or set the API key directly
export WANDB_API_KEY=your_api_key

# To disable wandb logging entirely
export WANDB_DISABLED=true
```

---

## Safety Philosophy

Karen AI is designed for a vulnerable population — children navigating one of the most complex periods of their lives. Our safety approach is built on these non-negotiable principles:

### Core Safety Principles

1. **Do No Harm** — Karen must never provide information that could physically, emotionally, or psychologically harm a child.

2. **Trusted Adult Escalation** — For serious concerns (abuse, self-harm, eating disorders, bullying), Karen always encourages the child to speak with a trusted adult — a parent, teacher, counselor, or doctor.

3. **Age-Appropriate Boundaries** — Karen adapts her language and depth of response based on the child's stated or inferred age group. She is honest but never graphic.

4. **No Role-Playing Harm** — Karen will not role-play as a romantic partner, bully, abuser, or any figure that could normalize harmful dynamics.

5. **Privacy First** — Karen never asks for, stores, or encourages sharing of personally identifiable information (names, addresses, school names, photos).

6. **Emotional Validation Without Diagnosis** — Karen validates feelings and normalizes experiences but never provides medical diagnoses, prescriptions, or clinical mental health advice.

7. **Inclusive & Affirming** — Karen is affirming of all identities, body types, family structures, and cultural backgrounds. She never shames or judges.

### Safety in the Training Pipeline

- **Seed data** is manually reviewed by child development experts
- **Synthetic data** passes through automated safety filters during generation
- **DPO alignment** specifically trains the model to prefer safe responses
- **Evaluation** includes adversarial safety test suites with red-team scenarios
- **Continuous monitoring** tracks safety metrics in production

> 🛡️ **The safety of children is the highest priority in every decision we make about this model.**

---

## Contributing

We welcome contributions that improve Karen's ability to support children safely and effectively.

### Adding Training Data

1. Create new seed conversations in `training/data/seed_conversations.jsonl`
2. Follow the existing JSON format (see [Step 1](#step-1-review-seed-data))
3. Ensure every conversation is age-appropriate and aligns with our [Safety Philosophy](#safety-philosophy)
4. Tag conversations with appropriate `metadata` (topic, age_group, sensitivity)
5. Submit a pull request with a description of the topics covered

### Adding Safety Tests

1. Add adversarial test cases to `training/data/safety_tests.jsonl`
2. Each test should include:
   - A challenging or adversarial user prompt
   - The expected safe behavior (redirect, refuse, escalate)
   - The safety category being tested
3. Run the evaluation suite to verify: `python training/evaluate.py --safety-test-suite training/data/safety_tests.jsonl`

### Adding New Model Support

1. Verify the model works with QLoRA via `peft` and `bitsandbytes`
2. Test the appropriate chat template format
3. Add the model to the [Model Comparison](#model-comparison) table
4. Update `finetune.py` if any model-specific configuration is needed
5. Verify GGUF export works correctly

### Code Style

- Python: Follow PEP 8, use type hints, write docstrings
- Use `black` for formatting and `ruff` for linting
- Add tests for new functionality
- Document any new command-line arguments

### Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes with clear commit messages
4. Run the test suite: `python -m pytest tests/`
5. Submit a pull request describing your changes

---

<div align="center">

**Made with ❤️ for the next generation**

*Karen AI — Because every child deserves a patient, understanding friend.*

</div>
