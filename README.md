# my-voice

Fine-tune a Llama 3.1 8B model on your personal texting style using Instagram DM exports, MLX LoRA, and LM Studio for data filtering.

This repo contains the **pipeline scripts only**. Raw chats, processed databanks, training JSONL, and adapter weights stay local and are not committed.

## Prerequisites

- macOS with Apple Silicon (MLX)
- Python 3.10+
- [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) for LoRA training and inference
- [LM Studio](https://lmstudio.ai/) running locally for the filtering step (`filter_databank.py` uses the OpenAI-compatible API at `http://127.0.0.1:1234/v1`)
- Base model: [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/mlx-community/Meta-Llama-3.1-8B-Instruct-8bit) (MLX format)

Install dependencies:

```bash
pip install -r requirements.txt
```

Set the base model path if yours differs from the default:

```bash
export MODEL="$HOME/.lmstudio/models/mlx-community/Meta-Llama-3.1-8B-Instruct-8bit"
```

## Folder structure

```
my-voice/
├── insta/                  # Raw Instagram DM exports (local only)
├── private_data/           # Processed databank + filtered messages (local only)
├── public_data/            # train.jsonl + valid.jsonl (local only)
├── scripts/
│   ├── build_databank.py         # Step 1: raw JSON → databank
│   ├── filter_databank.py        # Step 2: filter via LM Studio
│   ├── build_initial_train.py    # Step 3a: build initial train/valid
│   ├── build_final_train.py      # Step 3b: refine into final format
│   ├── profanity_re              # (local only) regex patterns for filtering
│   ├── personal_re                 # (local only) regex patterns for filtering
│   └── main.py                   # Step 5: interactive inference
└── training/
    ├── trainingLoRA.sh           # Step 4: LoRA training
    ├── system_prompt.txt         # (local only) system prompt for inference
    ├── paths.sh                  # (local only) model/data paths
    └── adapters_01/              # Trained LoRA weights (local only)
```

## Pipeline

| Step | Command | Output |
|------|---------|--------|
| 1 | Export Instagram DMs into `insta/` | `message_*.json` per chat |
| 2 | `python scripts/build_databank.py` | `private_data/databank.jsonl` |
| 3 | Start LM Studio, then `python scripts/filter_databank.py` | `private_data/style_bank_hybrid_filtered.jsonl` |
| 4a | `python scripts/build_initial_train.py` | `initial_train/train.jsonl`, `initial_train/valid.jsonl` |
| 4b | `python scripts/build_final_train.py` | `public_data/train.jsonl`, `public_data/valid.jsonl` |
| 5 | `bash training/trainingLoRA.sh` | `training/adapters_01/` |
| 6 | `python scripts/main.py` | Interactive chat |

### Step 1: Export Instagram DMs

Use Instagram's data export tool. Place each chat folder under `insta/` so the layout looks like:

```
insta/
└── friendname_123456789/
    └── message_1.json
```

### Step 2: Build databank

```bash
python scripts/build_databank.py --me "YourName"
```

Reads all `message_*.json` files, skips reactions/attachments/URLs, and writes one JSONL record per message.

### Step 3: Filter messages

Start LM Studio with Llama 3.1 8B Instruct loaded and the local server enabled, then:

```bash
python scripts/filter_databank.py
```

Uses rule-based filters plus an LLM pass to keep English, style-useful messages and reject noise, PII, and inappropriate content.

### Step 4a: Build initial training data

```bash
python scripts/build_initial_train.py
```

Converts filtered messages into chat-format examples (system / user / assistant turns) and splits into train (90%) and valid (10%). Outputs to `initial_train/`.

### Step 4b: Build final training data

```bash
python scripts/build_final_train.py
```

Refines the initial train/valid sets by adding the system prompt and filtering by message length. Outputs to `public_data/`.

Example `train.jsonl` line:

```json
{"messages": [{"role": "system", "content": "You are chatting in Praneel's personal texting style..."}, {"role": "user", "content": "Friend says: yo you coming?"}, {"role": "assistant", "content": "bet i'll be there"}]}
```

### Step 5: Train LoRA

```bash
bash training/trainingLoRA.sh
```

Training uses `--mask-prompt` so loss is computed only on assistant (your) replies, not on the system prompt or friend's message. This focuses the adapter on learning your response style.

### Step 6: Run inference

```bash
python scripts/main.py
```

Loads the base model + LoRA adapter and generates a reply using `training/system_prompt.txt`.

## Local-only files (gitignored)

These files contain personal or project-specific data and are not committed. Create them locally as needed:

| File | Purpose |
|------|---------|
| `scripts/profanity_re` | Regex pattern for profanity detection in `filter_databank.py` |
| `scripts/personal_re` | Regex pattern for intimate content detection in `filter_databank.py` |
| `training/system_prompt.txt` | System prompt defining your texting persona for inference |
| `training/paths.sh` | Environment variables for model/data/adapter paths |

## Privacy

**Never commit** the following:

- `insta/` — raw DM exports with real names and conversations
- `private_data/` — processed messages and filter audit logs
- `public_data/` — training examples derived from your chats
- `*.safetensors` — trained adapter weights

These paths are listed in `.gitignore`. Before your first push, verify with:

```bash
git status
```

If any data was previously tracked, remove it from the index without deleting local files:

```bash
git rm -r --cached insta/ private_data/ public_data/ training/adapters_01/*.safetensors 2>/dev/null || true
```
