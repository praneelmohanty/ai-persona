# my-voice

My-voice allows you to fine-tune an MLX-based Llama 3.1 8B model on your personal texting style using Instagram DM data, MLX LoRA and LM Studio.

This repo ships with **pre-trained safetensor adapter weights compatible with MLX** so you can directly chat with a model that texts similar to me. You can also run the full pipeline to train your own.

---

## What is needed

- **Mac with Apple Silicon** (M Chipset) — MLX requires Apple Silicon
- **macOS 14.0+**
- **Python 3.10+**

| RAM Needed | Quant version |
|------------|---------------|
| **16 GB+ RAM** | 16-bit (Unquantised) |
| **~16 GB RAM** | 8-bit |
| **~8 GB RAM** | 4-bit |
| **~4 GB RAM** | 2-bit (not recommended)|

---

## Step 1: Install LM Studio

1. Go to [lmstudio.ai](https://lmstudio.ai/download) and download LM Studio
2. Open LM Studio and install it
3. In LM Studio, search for `Meta-Llama-3.1-8B-Instruct` (the 8-bit MLX version) or any other quantisation
4. Download it — it will be saved to `~/.lmstudio/models/mlx-community/Meta-Llama-3.1-8B-Instruct-8bit`
5. Go to the **Developer** tab (second on the left sidebar)
6. Click the **Start Server** toggle button — this runs at `http://127.0.0.1:1234/v1`
7. Keep LM Studio running in the background when using the filter script

---

## Step 2: Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `mlx-lm` — MLX LLM inference and LoRA training
- `openai` — used to talk to LM Studio's local server
- `python-dotenv` — loads path configuration from `paths.env`

---

## Step 3: Clone the repo

```bash
git clone https://github.com/praneelmohanty/my-voice.git
cd my-voice
```

---

## Step 4: Set up your paths

The scripts use a `paths.env` file to find your model, data, and output directories. Create one from the template:

```bash
cp pathExample.txt paths.env
```

Open `paths.env` and replace every `/Users/yourname/` with your actual paths. Here's what each variable does:

| Variable | What it points to |
|----------|-------------------|
| `PROJECT_ROOT` | Root of the cloned repo |
| `INSTA_DIR` | Where your Instagram DM exports live |
| `PRIVATE_DATA_DIR` | Where processed databanks go |
| `PUBLIC_DATA_DIR` | Where final training data goes |
| `INITIAL_TRAIN_DIR` | Where initial train/valid splits go |
| `TRAINING_DIR` | Where training scripts and adapters live |
| `MODEL_PATH` | Path to the Llama 3.1 8B model (usually in `~/.lmstudio/models/`) |
| `ADAPTER_PATH` | Path to the LoRA adapter weights (`training/adapters_01/`) |
| `SYSTEM_PROMPT_PATH` | Path to `training/system_prompt.txt` |
| `DATABANK_INPUT` | Output of Step 1 (raw databank) |
| `DATABANK_OUTPUT` | Parsed Instagram messages |
| `FILTERED_OUTPUT` | Messages that passed filtering |
| `REJECTS_OUTPUT` | Messages that were rejected |
| `AUDIT_OUTPUT` | Full audit log of filter decisions |
| `TRAIN_INITIAL` | Initial training split |
| `VALID_INITIAL` | Initial validation split |
| `TRAIN_FINAL` | Final training data |
| `VALID_FINAL` | Final validation data |
| `PROFANITY_RE` | Regex file for profanity detection (create locally) |
| `PERSONAL_RE` | Regex file for personal content detection (create locally) |
| `LM_STUDIO_URL` | LM Studio server URL (`http://127.0.0.1:1234/v1`) |
| `LM_STUDIO_MODEL` | Model name for LM Studio API (`llama-3.1-8b-instruct`) |

Also set up the shell paths for the training scripts:

```bash
cp training/pathExamples.txt training/paths.sh
```

Edit `training/paths.sh` with the same model, data, and adapter paths.

---

## Step 5: Download your Instagram data

1. Go to [instagram.com](https://www.instagram.com/) on a browser
2. Click on the three bars (bottom left) → **Settings**
3. Go to Accounts Centre
4. Click on **Your information and permissions** → **Export your information**
5. Click on **Create export**
6. Choose your Instagram profile
7. Choose **Export to device**
8. Go to **Customise Information** 
9. Select **Some of your information** → **Clear all**
10. Scroll down and tick **Messages** — nothing else needed
11. Choose **Date range: Last year**, **Format: JSON** and **Quality: High** (any quality works, however)
12. Click **Start export**
13. Wait for the email from Instagram (can take minutes to hours)
14. Download the `.zip` file and unzip it
15. Inside you'll find a `messages/inbox/` folder with chat folders like:

    ```
    messages/inbox/
    ├── friendname_abc123/
    │   ├── message_1.json
    │   └── photos/
    ├── anotherfriend_def456/
    │   ├── message_1.json
    │   └── photos/
    └── ...
    ```
16. Copy each chat folder into `insta/` in this repo:
    ```
    my-voice/insta/
    ├── friendname_abc123/
    │   └── message_1.json
    └── anotherfriend_def456/
        └── message_1.json
    ```

---

## Folder structure

```
my-voice/
├── insta/                          # Your Instagram DM exports (gitignored)
├── private_data/                   # Processed data (gitignored)
├── public_data/                    # Final training data (gitignored)
├── initial_train/                  # Initial train/valid splits (gitignored)
├── paths.env                       # Your local paths (gitignored)
├── pathExample.txt                 # Template for paths.env
├── requirements.txt                # Python dependencies
├── scripts/
│   ├── build_databank.py           # Parses raw Instagram JSON
│   ├── filter_databank.py          # Filters messages via rules + LLM
│   ├── build_initial_train.py      # Builds train/valid splits
│   ├── build_final_train.py        # Refines into final format
│   ├── main.py                     # Interactive chat (Python)
│   ├── profanity_re                # (local) profanity regex (gitignored)
│   └── personal_re                 # (local) personal content regex (gitignored)
└── training/
    ├── trainingLoRA.sh             # LoRA training script
    ├── generateModel.sh            # Quick generation script
    ├── paths.sh                    # (local) shell paths (gitignored)
    ├── pathExamples.txt            # Template for paths.sh
    ├── system_prompt.txt           # (local) your persona prompt
    └── adapters_01/                # Pre-trained LoRA weights
        ├── adapter_config.json     # Generated settings of adapter (gitignored)
        ├── adapters.safetensors
        └── 0000200_adapters.safetensors
```

---

## Use the pre-trained model

This repo includes pre-trained LoRA adapter weights trained on my texting style. You can use them immediately.

### Option A: Python interactive chat (`main.py`)

```bash
python scripts/main.py
```

This will:
1. Load the base Llama 3.1 8B model from LM Studio
2. Load the LoRA adapter from `training/adapters_01/`
3. Load the system prompt from `training/system_prompt.txt`
4. Prompt you to enter a message
5. Generate a response in Praneel's texting style

Example session:
```
Enter a message: yo you coming tonight?
Praneel: bet i'll be there
```

### Option B: Shell script (`generateModel.sh`)

```bash
bash training/generateModel.sh
```

This runs `mlx_lm.generate` directly with the adapter loaded. Edit the `--prompt` flag in the script to change the input message.

### Option C: One-liner in terminal

You can also run `mlx_lm.generate` directly:

```bash
mlx_lm.generate \
  --model "/Users/yourname/.lmstudio/models/mlx-community/Meta-Llama-3.1-8B-Instruct-8bit" \
  --adapter-path "/Users/yourname/my-voice/training/adapters_01" \
  --system-prompt "$(cat /Users/yourname/my-voice/training/system_prompt.txt)" \
  --prompt "yo what's up"
```

Change `--prompt` to whatever message you want to test.

---

## Train your own model

If you want to train on your own texting style, run the full pipeline:

### Step 1: Build databank

```bash
python scripts/build_databank.py --me "YourName"
```

Replace `"YourName"` with your name exactly as it appears in Instagram exports. This reads all `message_*.json` files from `insta/`, skips reactions/attachments/URLs, and writes one JSONL record per message to `private_data/databank.jsonl`.

### Step 2: Filter messages

Make sure LM Studio is running with the Llama 3.1 8B model loaded and the local server started, then:

```bash
python scripts/filter_databank.py
```

Uses rule-based filters plus an LLM pass to keep English, style-useful messages and reject noise, PII, Hindi/Hinglish, and inappropriate content. Outputs:
- `private_data/style_bank_hybrid_filtered.jsonl` — kept messages
- `private_data/style_bank_hybrid_rejects.jsonl` — rejected messages
- `private_data/style_bank_hybrid_audit.jsonl` — full audit log

### Step 3a: Build initial training data

```bash
python scripts/build_initial_train.py
```

Converts filtered messages into chat-format examples (system / user / assistant turns) and splits into train (90%), valid (9%) and examples (50 lines). Outputs to `initial_train/`.

### Step 3b: Build final training data

```bash
python scripts/build_final_train.py
```

Refines the initial train/valid sets by adding the system prompt and filtering by message length. Outputs to `public_data/`.

### Step 4: Train LoRA

```bash
bash training/trainingLoRA.sh
```

This runs `mlx_lm.lora` with these parameters:
- **Model:** Llama 3.1 8B Instruct (8-bit)
- **Iterations:** 400
- **Batch size:** 4
- **Learning rate:** 1e-5
- **Steps per report:** 10
- **Steps per eval:** 50
- **Save every:** 100
- **Val batches:** 25
- **Mask prompt:** enabled (loss computed only on your replies, not the system prompt or friend's message)

Training takes ~15-20 minutes on an M4 Mac. Adapter weights are saved to `training/adapters_01/`. You can terminate the training anytime if the train loss does not go down using ^C. Try increasing or decreasing the parameters until the loss is able to drop to ~0.5-0.9

### Step 5: Chat with your model

```bash
python scripts/main.py
```

---

## Local-only files

These files are gitignored and stay on your machine:

| File | Purpose |
|------|---------|
| `paths.env` | Your local path configuration |
| `training/paths.sh` | Shell path variables for training scripts |
| `training/system_prompt.txt` | Your persona/system prompt |
| `training/adapters_01/adapter_config.json` | Adapter settings |
| `scripts/profanity_re` | Regex pattern for profanity filtering |
| `scripts/personal_re` | Regex pattern for personal content filtering |
| `insta/` | Raw Instagram DM exports |
| `private_data/` | Processed databanks and filter logs |
| `public_data/` | Final training data |
| `initial_train/` | Initial train/valid splits |

These are all listed in `.gitignore`. Before your first push, verify with:

```bash
git status
```

If any data was previously tracked, remove it from the index without deleting local files:

```bash
git rm -r --cached insta/ private_data/ public_data/ paths.env training/paths.sh training/system_prompt.txt 2>/dev/null || true
```

---

## License

Personal project. Do not redistribute training data or adapter weights without permission.