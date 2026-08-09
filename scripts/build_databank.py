from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_ME = "Praneel"
DEFAULT_INPUT_ROOT = Path("/Users/praneel/Desktop/AI Learning/praneel-voice/my-voice/insta")
DEFAULT_OUTPUT_PATH = Path("/Users/praneel/Desktop/AI Learning/praneel-voice/my-voice/private_data/instagram_data_bank.jsonl")

SKIP_EXACT = {
    "liked a message",
}

SKIP_PREFIXES = (
    "reacted ",
)

ATTACHMENT_MARKERS = (
    "sent an attachment.",
)

WHITESPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")


def clean_text(text: str) -> str:
    text = text.replace("\u200e", " ").replace("\u202f", " ")
    text = text.replace("\r", "\n")
    text = text.replace("\x00", " ")
    text = text.strip()
    lines = [WHITESPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def should_skip_message(msg: dict[str, Any]) -> bool:
    if msg.get("photos") or msg.get("videos") or msg.get("audio_files") or msg.get("gifs") or msg.get("share"):
        return True

    content = msg.get("content")
    if not isinstance(content, str):
        return True

    text = clean_text(content)
    if not text:
        return True

    lowered = text.lower()
    if lowered in SKIP_EXACT:
        return True
    if lowered.startswith(SKIP_PREFIXES):
        return True
    if any(marker in lowered for marker in ATTACHMENT_MARKERS):
        return True
    if URL_RE.search(text):
        return True

    return False


def parse_message_file(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def collect_json_files(chat_dir: Path) -> list[Path]:
    files = []
    for candidate in sorted(chat_dir.glob("message_*.json")):
        if candidate.is_file():
            files.append(candidate)
    return files


def normalize_chat(chat_dir: Path, me_name: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []

    for json_file in collect_json_files(chat_dir):
        payload = parse_message_file(json_file)
        if not payload:
            continue

        title = payload.get("title") or chat_dir.name
        thread_path = payload.get("thread_path", "")

        for msg in payload.get("messages", []):
            if should_skip_message(msg):
                continue

            sender = clean_text(str(msg.get("sender_name", "")))
            text = clean_text(msg["content"])
            ts_ms = msg.get("timestamp_ms")

            if not sender or not text or not isinstance(ts_ms, int):
                continue

            role = "Praneel" if sender == me_name else sender

            normalized.append(
                {
                    "source": "instagram",
                    "chat_title": title,
                    "thread_path": thread_path,
                    "sender": role,
                    "is_me": sender == me_name,
                    "timestamp_ms": ts_ms,
                    "timestamp_iso": datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat(),
                    "text": text,
                    "file": json_file.name,
                }
            )

    normalized.sort(key=lambda item: item["timestamp_ms"])
    return normalized


def build_records(chat_dir: Path, me_name: str) -> list[dict[str, Any]]:
    messages = normalize_chat(chat_dir, me_name)
    records: list[dict[str, Any]] = []

    for idx, message in enumerate(messages, start=1):
        records.append(
            {
                "source": "instagram",
                "chat_folder": chat_dir.name,
                "chat_title": message["chat_title"],
                "thread_path": message["thread_path"],
                "sequence": idx,
                "sender": message["sender"],
                "text": message["text"],
                "timestamp_ms": message["timestamp_ms"],
                "timestamp_iso": message["timestamp_iso"],
                "file": message["file"],
            }
        )

    return records


def iter_chat_dirs(input_root: Path) -> list[Path]:
    if not input_root.exists():
        return []
    return sorted(path for path in input_root.iterdir() if path.is_dir())


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an Instagram-only conversational data bank from exported chats."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root folder containing Instagram chat folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSONL path for the conversational data bank.",
    )
    parser.add_argument(
        "--me",
        default=DEFAULT_ME,
        help="Your sender name exactly as it appears in Instagram exports.",
    )
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    chat_dirs = iter_chat_dirs(args.input_root)

    for chat_dir in chat_dirs:
        all_rows.extend(build_records(chat_dir, args.me))

    write_jsonl(all_rows, args.output)

    print(f"Chats processed: {len(chat_dirs)}")
    print(f"Messages written: {len(all_rows)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
