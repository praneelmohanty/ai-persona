#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

BASE_DIR = Path("/Users/praneel/Desktop/AI Learning/praneel-voice/my-voice")
INPUT_PATH = BASE_DIR / "private_data" / "style_bank_hybrid_filtered.jsonl"
OUTPUT_DIR = BASE_DIR / "initial_train"
TRAIN_OUT = OUTPUT_DIR / "train.jsonl"
VALID_OUT = OUTPUT_DIR / "valid.jsonl"
PREVIEW_OUT = OUTPUT_DIR / "examples_preview.jsonl"

SYSTEM_PROMPT = (
    "You are Praneel texting casually with friends."
    "Style: lowercase, short, slang (lmao, fr, bet, ong, fam, rlly, tbh, mb, ik, dead, weak, lowk). Minimal punctuation. Hinglish mix natural."
)

SEED = 42
HOLDOUT_RATIO = 0.1
MAX_TURNS_PER_CHAT = 100000

def normalize(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\u200e", "").replace("\u202f", "").replace("\u0000", "")
    return " ".join(text.split()).strip()

def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows

def get_chat_key(row: dict[str, Any]) -> tuple[str, str]:
    source = normalize(row.get("source", "unknown"))
    thread_path = normalize(row.get("thread_path", ""))
    chat_folder = normalize(row.get("chat_folder", ""))
    chat_title = normalize(row.get("chat_title", "unknown"))

    if thread_path:
        return (source, thread_path)
    if chat_folder:
        return (source, chat_folder)
    return (source, chat_title)

def group_same_sender(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    grouped = []

    for row in messages:
        sender = normalize(row.get("sender", ""))
        text = normalize(row.get("text", ""))

        if not grouped:
            grouped.append({"sender": sender, "text": text})
            continue

        if grouped[-1]["sender"] == sender:
            if grouped[-1]["text"]:
                grouped[-1]["text"] += "\n" + text
            else:
                grouped[-1]["text"] = text
        else:
            grouped.append({"sender": sender, "text": text})

    return grouped

def build_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chats: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        chats[get_chat_key(row)].append(row)

    examples = []

    for _, msgs in chats.items():
        msgs.sort(key=lambda r: (r.get("timestamp_ms", 0), r.get("sequence", 0)))
        grouped = group_same_sender(msgs)

        turns_added = 0
        for i in range(len(grouped) - 1):
            first = grouped[i]
            second = grouped[i + 1]

            if first["sender"] == second["sender"]:
                continue

            user_text = normalize(first["text"])
            assistant_text = normalize(second["text"])

            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ]
            }
            examples.append(example)
            turns_added += 1

            if turns_added >= MAX_TURNS_PER_CHAT:
                break

    return examples

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def main() -> None:
    rows = load_rows(INPUT_PATH)
    examples = build_examples(rows)

    random.seed(SEED)
    random.shuffle(examples)

    valid_count = int(len(examples) * HOLDOUT_RATIO)
    if len(examples) > 0 and valid_count == 0:
        valid_count = 1

    valid = examples[:valid_count]
    train = examples[valid_count:]
    preview = examples[:50]

    write_jsonl(TRAIN_OUT, train)
    write_jsonl(VALID_OUT, valid)
    write_jsonl(PREVIEW_OUT, preview)

    stats = {
        "input_path": str(INPUT_PATH),
        "rows_loaded": len(rows),
        "examples_total": len(examples),
        "train": len(train),
        "valid": len(valid),
        "preview": len(preview),
        "train_out": str(TRAIN_OUT),
        "valid_out": str(VALID_OUT),
        "preview_out": str(PREVIEW_OUT),
    }
    print(json.dumps(stats, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
