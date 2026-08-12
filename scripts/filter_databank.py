from __future__ import annotations
import argparse
import json
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any
from openai import OpenAI
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / "paths.env")

INPUT_PATH = Path(os.getenv("DATABANK_INPUT"))
OUTPUT_KEEP = Path(os.getenv("FILTERED_OUTPUT"))
OUTPUT_REJECT = Path(os.getenv("REJECTS_OUTPUT"))
OUTPUT_AUDIT = Path(os.getenv("AUDIT_OUTPUT"))

MODEL = os.getenv("LM_STUDIO_MODEL", "llama-3.1-8b-instruct")
client = OpenAI(base_url=os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234/v1"), api_key="lmstudio")

URL_RE = re.compile(r"(?:https?://\S+|www\.\S+|\b[a-z0-9.-]+\.(?:com|org|net|edu|gov|io|gg|tv|me|co|in)(?:/\S*)?\b)", re.I)
TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
ALPHA_RE = re.compile(r"[A-Za-z]")
ONLY_PUNCT_RE = re.compile(r"^[\W_]+$")
MULTISPACE_RE = re.compile(r"\s+")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
HANDLE_RE = re.compile(r"(?<!\w)@[a-z0-9._]{2,32}\b", re.I)
LONG_NUMBER_RE = re.compile(r"\b\d{7,}\b")
DOB_RE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s*\d{2,4})?)\b", re.I)
REPEATED_CHAR_RE = re.compile(r"(.)\1{5,}")
WORD_SALAD_RE = re.compile(r"\b[A-Za-z]{10,}\b")

MEDIA_PLACEHOLDERS = {
    "image omitted", "video omitted", "audio omitted", "document omitted", "sticker omitted", "gif omitted",
    "video note omitted", "you sent an attachment.", "you sent a sticker.", "you deleted this message.",
    "sent a photo", "sent a video", "sent an audio", "shared a reel", "shared a post"
}
SYSTEM_PATTERNS = [
    "messages and calls are end-to-end encrypted", "changed their phone number", "missed voice call",
    "missed video call", "started an audio call", "started a video call", "this message was deleted",
    "answered on other device", "click to call back", "changed the theme", "reacted to your message",
    "unsent a message", "liked a message", "replied to the story", "shared a story"
]
SAFE_EXPRESSIVE = {
    "real", "so real", "for real", "lol", "lmao", "lmaoo", "lmaooo", "haha", "hahaha", "hehe", "heheh",
    "omg", "omgg", "omggg", "bro", "broski", "man", "lowk", "highkey", "ong", "goated", "okie", "okiee",
    "yess", "yesss", "yay", "aw", "aww", "awww", "awwww", "damn", "nice", "okay", "okayy",
    "facts", "crazy", "wild", "true", "same", "sameee", "sameeee", "fr", "rn", "idk", "tbh", "ngl"
}
PROFANITY_PATTERN = Path(os.getenv("PROFANITY_RE")).read_text().strip() #Write a RegEx flags to ignore any profanity in your chats
PROFANITY_RE = re.compile(PROFANITY_PATTERN, re.IGNORECASE)

PERSONAL_PATTERN = Path(os.getenv("PERSONAL_RE")).read_text().strip() #Write a RegEx flags to ignore any personal information in your chats
PERSONAL_RE = re.compile(PERSONAL_PATTERN, re.IGNORECASE)

ENGLISH_WORDS = {
    "a","about","actually","after","again","all","also","am","an","and","any","are","around","as","at","back",
    "bad","be","because","been","before","being","best","better","but","by","call","can","cant","come","coming",
    "course","credits","day","design","did","didnt","do","does","doing","dont","driving","dude","dw","email",
    "entire","even","exam","fight","flight","for","from","get","gift","give","go","going","gonna","good","got",
    "great","guess","had","happened","happy","has","have","he","hell","her","here","hey","him","his","home",
    "hope","how","i","idk","if","ill","im","in","into","is","it","its","ive","jersey","just","keep","know",
    "land","legit","like","likes","look","lot","major","mall","man","math","maybe","me","mean","message","more",
    "most","my","need","net","nice","night","no","not","now","number","of","okay","one","only","or","out",
    "past","pay","pdf","people","pick","picking","place","player","print","printed","price","proud","really",
    "reaching","right","safe","said","school","search","see","send","sent","she","sleep","slow","so","some",
    "something","sounds","still","stuff","support","sure","talk","team","text","than","that","the","their","them",
    "then","there","they","thing","think","this","tho","though","time","title","to","today","too","travel","trust",
    "trying","up","us","use","very","watch","way","we","well","went","were","what","when","where","which","who",
    "why","will","win","with","work","world","would","wrong","yeah","yes","you","your","youre","yours","tbh",
    "ngl","ts","fr","rn","fifa","football","soccer","ferrari","mclaren","max","lando","messi","ronaldo","argentina",
    "chelsea","spurs","barca","madrid","valverde","bellingham","yamal","mbappe","haaland","cs","physics","econ","uni",
    "college","class","discussion","hall","lab","website","coding","study","ticket","boston","delhi","mumbai","indore",
    "columbus","holyoke","umass","attendance","downloads","book","parents","opinion","satire","choice","interest"
}

HINDI_MARKERS = {
    "acha","achha","accha","arey","arre","bhai","bhy","kya","kyu","kyun","kaise","nahi","haan","han","mera","meri",
    "mere","tera","teri","tum","tu","main","mei","mein","mujhe","kar","karo","karna","kr","krra","krri","chal",
    "gaya","gayi","aaya","aayi","abhi","kal","ghar","samajh","pata","bolo","dekh","thoda","zyada","bahut","bohot",
    "yaar","theek","sahi","wala","wali","wale","sab","kaun","kab","tak","phir","toh","na","aur","bhi","sirf",
    "lekin","par","agar","hum","hume","dono","yaad","jana","aana","suno","bata","batana","bhej","mil","milna",
    "dena","diya","do","hai","tha","thi","hua","ho","hoga","hogi","rah","raha","rha","kuch","firse","itna",
    "itni","fir","baar","aisa","waise","uska","uske","iska","iske","idhar","udhar","waha","yaha","rahe","rahi",
    "jaunga","jaungi","padhai","kaisa","bbsr","wakt","waqt","subah","raat","mazze","mast","hein","cheee","chup"
}

PERSONA_PATTERNS = [
    re.compile(r"\b(i think|i feel|i guess|i hope|i swear|i know|i mean|i dont think|i don['’]t think)\b", re.I),
    re.compile(r"\b(what do you think|why would|how could|sounds good|for real|so real)\b", re.I),
    re.compile(r"\b(are you|do you|did you|have you|will you|what are you|where are you)\b", re.I),
    re.compile(r"\b(gonna|wanna|gotta|idk|tbh|ngl|lowk|bro|dude|man|damn)\b", re.I),
]

SYSTEM_PROMPT = """You are filtering chat messages for a personal style-training dataset.
Return only valid JSON.
Rules:
- KEEP only if the message is primarily English and useful for modeling casual text style.
- REJECT Hindi, Hinglish, intimate or sexual content, explicit content, personal info, gibberish, timestamps, IDs, low-signal acknowledgements, media/system text, and mojibake-only junk.
- Messages with obvious PII should be rejected, not redacted.
- Prefer precision over recall.
Output schema:
{"decision":"KEEP"|"REJECT","reason":"short_reason"}
"""

MOJIBAKE_FIXES = {
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€": '"',
    "â€\x9d": '"',
    "â€\x9c": '"',
    "â€“": "-",
    "â€”": "-",
    "â€¦": "...",
    "Â": "",
    "Ã©": "é",
}

def safe_json_loads(s: str) -> dict[str, Any] | None:
    try:
        return json.loads(s)
    except Exception:
        return None

def fix_common_mojibake(text: str) -> str:
    out = text
    for bad, good in MOJIBAKE_FIXES.items():
        out = out.replace(bad, good)
    try:
        if any(ch in out for ch in "ÃÂâ"):
            repaired = out.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
            if repaired and repaired.count("�") <= out.count("�"):
                out = repaired
    except Exception:
        pass
    return out

def strip_invisible_and_control(text: str) -> str:
    cleaned = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in "\n\t ":
            cleaned.append(ch)
        elif cat.startswith("C"):
            continue
        else:
            cleaned.append(ch)
    return "".join(cleaned)

def normalize_text(text: str) -> str:
    text = str(text or "")
    text = fix_common_mojibake(text)
    text = strip_invisible_and_control(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200e", " ").replace("\u202f", " ").replace("\xa0", " ")
    text = text.replace("’", "'").replace("`", "'")
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text

def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())

def english_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in ENGLISH_WORDS) / len(tokens)

def hindi_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in HINDI_MARKERS) / len(tokens)

def looks_like_system(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in SYSTEM_PATTERNS)

def has_private_info(text: str) -> bool:
    if EMAIL_RE.search(text):
        return True
    if HANDLE_RE.search(text):
        return True
    if PHONE_RE.search(text):
        return True
    if LONG_NUMBER_RE.search(text):
        return True
    if DOB_RE.search(text) and len(text) < 40:
        return True
    return False

def is_mostly_noise(text: str) -> tuple[bool, str | None]:
    if not text:
        return True, "empty"
    if text.lower() in MEDIA_PLACEHOLDERS:
        return True, "media_placeholder"
    if looks_like_system(text):
        return True, "system_message"
    if URL_RE.fullmatch(text.lower()):
        return True, "url_only"
    if ONLY_PUNCT_RE.fullmatch(text):
        return True, "punct_only"
    if len(ALPHA_RE.findall(text)) < 2 and not any(ch.isdigit() for ch in text):
        return True, "non_text"
    if REPEATED_CHAR_RE.search(text.lower()) and len(text) < 8:
        return True, "char_spam"
    return False, None

def is_low_signal(tokens: list[str], text: str) -> tuple[bool, str | None]:
    lo = text.lower()
    if lo in {"ok", "okay", "kk", "k", "hmm", "hm", "oh", "ohh", "ohhh", "yes", "no", "nah", "yup", "nope"}:
        return True, "low_signal_ack"
    if text.isdigit() and len(text) <= 4:
        return True, "number_only"
    if len(tokens) <= 1 and lo not in SAFE_EXPRESSIVE:
        return True, "too_short"
    if len(tokens) <= 2 and len(text) <= 6 and lo not in SAFE_EXPRESSIVE:
        return True, "too_short"
    if WORD_SALAD_RE.search(text) and english_ratio(tokens) < 0.34:
        return True, "gibberish"
    return False, None

def is_obvious_keep(text: str) -> tuple[bool, str | None]:
    t = normalize_text(text)
    lo = t.lower()
    toks = tokenize(t)
    if lo in SAFE_EXPRESSIVE:
        return True, "safe_expressive"
    if len(toks) >= 4 and hindi_ratio(toks) == 0 and english_ratio(toks) >= 0.8:
        if any(p.search(t) for p in PERSONA_PATTERNS) or len(t) >= 18:
            return True, "clear_english_persona"
    if len(toks) >= 6 and hindi_ratio(toks) == 0 and english_ratio(toks) >= 0.72:
        return True, "likely_clean_english"
    return False, None

def is_obvious_reject(text: str) -> tuple[bool, str | None]:
    t = normalize_text(text)
    toks = tokenize(t)
    noise, reason = is_mostly_noise(t)
    if noise:
        return True, reason
    if has_private_info(t):
        return True, "personal_info"
    if PROFANITY_RE.search(t):
        return True, "profanity"
    if PERSONAL_RE.search(t):
        return True, "sexual_or_intimate"
    if hindi_ratio(toks) > 0:
        return True, "contains_hindi_or_hinglish"
    if len(toks) > 0 and english_ratio(toks) < 0.35:
        return True, "not_english_enough"
    low, low_reason = is_low_signal(toks, t)
    if low:
        return True, low_reason
    return False, None

def llm_classify(text: str) -> dict[str, str]:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Message: {text}\nClassify this message and return JSON only."},
        ],
    )
    content = (response.choices[0].message.content or "").strip()
    try:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1:
            return {"decision": "REJECT", "reason": "llm_parse_error"}
        parsed = json.loads(content[start:end + 1])
        decision = parsed.get("decision", "REJECT")
        reason = parsed.get("reason", "llm_no_reason")
        if decision not in {"KEEP", "REJECT"}:
            return {"decision": "REJECT", "reason": "llm_invalid_decision"}
        return {"decision": decision, "reason": reason}
    except Exception:
        return {"decision": "REJECT", "reason": "llm_parse_error"}

def process_line(obj: dict[str, Any], line_no: int) -> tuple[str, str, dict[str, Any]]:
    text = normalize_text(obj.get("text", ""))
    obj = dict(obj)
    obj["text"] = text

    reject, reject_reason = is_obvious_reject(text)
    if reject:
        return "REJECT", reject_reason or "reject", obj

    keep, keep_reason = is_obvious_keep(text)
    if keep:
        return "KEEP", keep_reason or "keep", obj

    verdict = llm_classify(text)
    return verdict["decision"], verdict["reason"], obj

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--keep", type=Path, default=OUTPUT_KEEP)
    parser.add_argument("--reject", type=Path, default=OUTPUT_REJECT)
    parser.add_argument("--audit", type=Path, default=OUTPUT_AUDIT)
    args = parser.parse_args()

    kept = 0
    rejected = 0
    llm_used = 0
    reasons = Counter()
    args.keep.parent.mkdir(parents=True, exist_ok=True)

    with args.input.open("r", encoding="utf-8") as fin, \
         args.keep.open("w", encoding="utf-8") as fkeep, \
         args.reject.open("w", encoding="utf-8") as frej, \
         args.audit.open("w", encoding="utf-8") as faudit:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            obj = safe_json_loads(line)
            if obj is None:
                reasons["invalid_json"] += 1
                frej.write(json.dumps({"line_no": line_no, "reason": "invalid_json", "raw": line}, ensure_ascii=False) + "\n")
                rejected += 1
                continue

            text = normalize_text(obj.get("text", ""))
            obvious_reject, obvious_reject_reason = is_obvious_reject(text)
            obvious_keep, obvious_keep_reason = (False, None) if obvious_reject else is_obvious_keep(text)
            used_llm = not obvious_reject and not obvious_keep
            if used_llm:
                llm_used += 1

            decision, reason, cleaned_obj = process_line(obj, line_no)
            reasons[reason] += 1
            faudit.write(json.dumps({
                "line_no": line_no,
                "decision": decision,
                "reason": reason,
                "used_llm": used_llm,
                "text": cleaned_obj.get("text", "")
            }, ensure_ascii=False) + "\n")

            if decision == "KEEP":
                fkeep.write(json.dumps(cleaned_obj, ensure_ascii=False) + "\n")
                kept += 1
            else:
                frej.write(json.dumps({"line_no": line_no, "reason": reason, "item": cleaned_obj}, ensure_ascii=False) + "\n")
                rejected += 1

    print(json.dumps({
        "kept": kept,
        "rejected": rejected,
        "llm_used": llm_used,
        "top_reasons": reasons.most_common(20),
        "model": MODEL,
        "output_keep": str(args.keep),
        "output_reject": str(args.reject),
        "output_audit": str(args.audit),
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()