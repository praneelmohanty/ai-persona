import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / "paths.env")

SYSTEM_PROMPT = (
    "You are Praneel texting casually with friends. "
    "Style: lowercase, short, natural reactions. Use slang/abbreviations "
    "(lmao, ik, fr, mb, fam, rlly, tbh, ong, bet, dead, weak). "
    "Minimal punctuation. One or two short sentences max. "
    "Match the energy — hype, roasting, venting, making plans, "
    "random observations, memes. No formal grammar."
)

def clean_text(text):
    """Basic cleanup of extracted messages."""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def is_your_message(role):
    """In your data, 'assistant' = your messages."""
    return role == "assistant"

def extract_pairs(input_path, output_path):
    with open(input_path) as f:
        lines = [json.loads(l) for l in f]

    pairs = []
    for entry in lines:
        msgs = entry["messages"]
        for i in range(1, len(msgs)):
            prev = msgs[i - 1]
            curr = msgs[i]
            if is_your_message(curr["role"]) and prev["role"] == "user":
                user_text = clean_text(prev["content"])
                asst_text = clean_text(curr["content"])
                if len(asst_text) < 10 or len(asst_text) > 400:
                    continue
                if asst_text.lower() in {"fr", "lmao", "lol", "ok", "yeah", "idk", "bet", "ong"}:
                    continue
                pairs.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Friend says: {user_text}"},
                        {"role": "assistant", "content": asst_text}
                    ]
                })

    with open(output_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print(f"Converted {len(lines)} raw conversations → {len(pairs)} clean pairs")
    return pairs

if __name__ == "__main__":
    extract_pairs(
        os.getenv("TRAIN_INITIAL"),
        os.getenv("TRAIN_FINAL")
    )
    extract_pairs(
        os.getenv("VALID_INITIAL"),
        os.getenv("VALID_FINAL")
    )